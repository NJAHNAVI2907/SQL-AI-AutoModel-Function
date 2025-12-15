# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION jahnavi.def.ai_ml_classify(source_table STRING, feature_list ARRAY<STRING>, target_col STRING, train BOOLEAN,feature_schema MAP<STRING, DOUBLE>)
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC ENVIRONMENT (
# MAGIC   dependencies = '["databricks-sql-connector","databricks-sdk","openai"]',
# MAGIC   environment_version = 'None'
# MAGIC )
# MAGIC AS $$
# MAGIC
# MAGIC from databricks import sql
# MAGIC import pandas as pd
# MAGIC import openai
# MAGIC from sklearn.model_selection import GridSearchCV
# MAGIC from sklearn.pipeline import Pipeline
# MAGIC from sklearn.preprocessing import StandardScaler
# MAGIC
# MAGIC from sklearn.linear_model import LogisticRegression
# MAGIC from sklearn.tree import DecisionTreeClassifier
# MAGIC from sklearn.naive_bayes import GaussianNB
# MAGIC from sklearn.metrics import accuracy_score
# MAGIC from joblib import dump,load
# MAGIC
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC import json
# MAGIC import numpy as np
# MAGIC import requests 
# MAGIC import os
# MAGIC import io
# MAGIC import re
# MAGIC
# MAGIC def build_model_path(source_table, feature_list, target_col):
# MAGIC     source_table_norm = re.sub(r"[^a-zA-Z0-9_]", "_", source_table.lower())
# MAGIC     target_col_norm = re.sub(r"[^a-zA-Z0-9_]", "_", target_col.lower())
# MAGIC
# MAGIC     normalized_features = [
# MAGIC         re.sub(r"[^a-zA-Z0-9_]", "_", f.lower()) for f in feature_list
# MAGIC     ]
# MAGIC     features_key = "_".join(sorted(normalized_features))
# MAGIC     model_key = f"{source_table_norm}__{target_col_norm}__{features_key}"
# MAGIC
# MAGIC     return f"/Volumes/jahnavi/def/ai_ml_classify/models/{model_key}.joblib"
# MAGIC
# MAGIC
# MAGIC ######################### GET SOURCE DATA #################################
# MAGIC
# MAGIC def getPandasData(server_hostname, http_path, access_token, table_name):
# MAGIC     
# MAGIC     conn = sql.connect(
# MAGIC         server_hostname=server_hostname,
# MAGIC         http_path=http_path,
# MAGIC         access_token=access_token
# MAGIC     )
# MAGIC     table_name = "main.yash.wine_training"
# MAGIC     query = f"SELECT * FROM {table_name}" 
# MAGIC
# MAGIC     with conn.cursor() as cursor:
# MAGIC         cursor.execute(query)
# MAGIC         result = cursor.fetchall()
# MAGIC         pdf = pd.DataFrame([r.asDict() for r in result])
# MAGIC     
# MAGIC     return pdf
# MAGIC     
# MAGIC ####################### EDA STATS ###################
# MAGIC
# MAGIC def transform(rows_json: str, feature_list: list, target_col: str) -> str:
# MAGIC     rows = json.loads(rows_json)
# MAGIC     feats = feature_list
# MAGIC     target = target_col
# MAGIC
# MAGIC     pdf = pd.DataFrame(rows)
# MAGIC
# MAGIC     cols = feats + ([target_col] if target_col in pdf.columns else [])
# MAGIC     pdf = pdf[cols].copy()
# MAGIC     eda_stats = {}
# MAGIC     eda_stats["feature_list"] = ",".join(feats)
# MAGIC     eda_stats["target_col"] = target_col
# MAGIC     eda_stats["sample_row_count"] = int(len(pdf))
# MAGIC     eda_stats["null_counts_sample"] = pdf.isna().sum().astype(int).to_dict()
# MAGIC
# MAGIC     num_cols = pdf.select_dtypes(include=[np.number]).columns.tolist()
# MAGIC     numeric_stats = {}
# MAGIC     if num_cols:
# MAGIC         desc = pdf[num_cols].describe()
# MAGIC         numeric_stats["count"] = {c: float(desc.loc["count", c]) for c in num_cols}
# MAGIC         numeric_stats["mean"] = {c: float(desc.loc["mean", c]) for c in num_cols}
# MAGIC         numeric_stats["stddev"] = {c: float(desc.loc["std", c]) for c in num_cols}
# MAGIC         numeric_stats["min"] = {c: float(desc.loc["min", c]) for c in num_cols}
# MAGIC         numeric_stats["max"] = {c: float(desc.loc["max", c]) for c in num_cols}
# MAGIC         eda_stats["numeric_stats_sample"] = numeric_stats
# MAGIC
# MAGIC     return eda_stats
# MAGIC
# MAGIC def build_column_stats(input_dict):
# MAGIC     feature_list = [c.strip() for c in input_dict["feature_list"].split(",")]
# MAGIC     target_col = input_dict["target_col"]
# MAGIC
# MAGIC     numeric = input_dict.get("numeric_stats_sample", {})
# MAGIC     nulls = input_dict.get("null_counts_sample", {})
# MAGIC     row_count = input_dict.get("sample_row_count", 0)
# MAGIC
# MAGIC     all_cols = feature_list + ([target_col] if target_col else [])
# MAGIC
# MAGIC     result = {}
# MAGIC
# MAGIC     for col in all_cols:
# MAGIC         count = numeric.get("count", {}).get(col)
# MAGIC         min_v = numeric.get("min", {}).get(col)
# MAGIC         max_v = numeric.get("max", {}).get(col)
# MAGIC         mean_v = numeric.get("mean", {}).get(col)
# MAGIC
# MAGIC         # infer dtype (simple heuristic)
# MAGIC         if min_v is not None and max_v is not None and float(min_v).is_integer() and float(max_v).is_integer():
# MAGIC             dtype = "bigint"
# MAGIC         else:
# MAGIC             dtype = "double"
# MAGIC
# MAGIC         null_count = nulls.get(col, None)
# MAGIC         null_pct = (null_count / row_count) * 100 if null_count is not None and row_count else None
# MAGIC
# MAGIC         # unique values only safely inferable for binary-like columns
# MAGIC         unique_values = None
# MAGIC         if dtype == "bigint" and min_v is not None and max_v is not None and max_v - min_v == 1:
# MAGIC             unique_values = [int(min_v), int(max_v)]
# MAGIC
# MAGIC         result[col] = {
# MAGIC             "dtype": dtype,
# MAGIC             "null_pct": null_pct,
# MAGIC             "unique_values": unique_values,
# MAGIC             "min": min_v,
# MAGIC             "max": max_v,
# MAGIC             "mean": mean_v
# MAGIC         }
# MAGIC
# MAGIC     return result
# MAGIC
# MAGIC ####################### MARDOWN FOR EDA ANALYSIS ###################
# MAGIC
# MAGIC def ask_gpt_for_eda(result):
# MAGIC     eda_prompt_stats = result
# MAGIC
# MAGIC     eda_prompt = f"""
# MAGIC     {{
# MAGIC         "feature_list": "{feature_list}",
# MAGIC         "target_col": "{target_col}",
# MAGIC         "row_count": 4527,
# MAGIC         "columns": {json.dumps(eda_prompt_stats)}
# MAGIC     }}
# MAGIC     """
# MAGIC
# MAGIC
# MAGIC     system_prompt_final = """
# MAGIC     You are an expert Machine Learning Engineer.
# MAGIC
# MAGIC     Given table metadata and EDA statistics for a tabular dataset, analyze it and respond in MARKDOWN ONLY with the following sections, in this exact order and format:
# MAGIC
# MAGIC     ### Problem Type
# MAGIC     - One bullet stating whether the task is **classification** or **regression**, with a brief justification.
# MAGIC
# MAGIC     ### Target Column
# MAGIC     - One bullet naming the target column.
# MAGIC
# MAGIC     ### Feature Columns
# MAGIC     - One bullet that starts with: `Feature columns:` followed by a comma-separated list of feature column names.
# MAGIC     - Do NOT include any columns listed under "Columns to Ignore".
# MAGIC
# MAGIC     ### Columns to Ignore
# MAGIC     - One bullet that starts with: `Columns to ignore:` followed by a comma-separated list.
# MAGIC     - Always include: alcohol, free_sulfur_dioxide, volatile_acidity.
# MAGIC
# MAGIC     ### Potential Issues
# MAGIC     - 3–5 bullets listing concrete data issues (missing values, imbalance, outliers, leakage, correlations, etc.).
# MAGIC     - Base these on the given metadata and stats only; if something is not directly supported, phrase it as a hypothesis (e.g. "Potential class imbalance (not verified from the stats).").
# MAGIC
# MAGIC     ### Recommended Models (from scikit-learn)
# MAGIC     - 2–5 bullets with specific sklearn estimator class names appropriate for the problem type, e.g. `LogisticRegression`, `RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostingRegressor`.
# MAGIC
# MAGIC     ### Dataset Statistics Summary
# MAGIC     - 4–5 bullets summarizing the EDA in a technically accurate way (row counts, ranges, distributions, nulls, target behavior, etc.).
# MAGIC     - Use only information that is consistent with the provided metadata and statistics.
# MAGIC
# MAGIC     Constraints:
# MAGIC     - Output MUST be valid markdown and MUST NOT contain any raw JSON, code fences, or extra sections.
# MAGIC     - Do NOT include any "Here is ..." or explanatory prose outside the defined sections.
# MAGIC     - Keep the entire response concise (max ~250 words).
# MAGIC     """
# MAGIC
# MAGIC     DATABRICKS_MODEL_URL = "https://cust-e2-us-west-2.cloud.databricks.com/serving-endpoints/databricks-gpt-5-1/invocations"
# MAGIC
# MAGIC     headers = {
# MAGIC         "Authorization": f"Bearer ",
# MAGIC         "Content-Type": "application/json",
# MAGIC     }
# MAGIC     payload = {
# MAGIC         "messages": [
# MAGIC             {
# MAGIC                 "role": "system",
# MAGIC                 "content": system_prompt_final
# MAGIC             },
# MAGIC             {
# MAGIC                 "role": "user",
# MAGIC                 "content": eda_prompt
# MAGIC             }
# MAGIC         ]
# MAGIC     }
# MAGIC
# MAGIC     resp = requests.post(DATABRICKS_MODEL_URL, headers=headers, json=payload)
# MAGIC     resp.raise_for_status()
# MAGIC     eda_mrkdown_text = resp.json()["choices"][0]["message"]["content"]
# MAGIC     buf = io.BytesIO()
# MAGIC     dump(eda_mrkdown_text, buf)
# MAGIC     buf.seek(0)
# MAGIC     host = "https://cust-e2-us-west-2.cloud.databricks.com"
# MAGIC     token = ""
# MAGIC     target = "/Volumes/jahnavi/def/ai_ml_classify/models/model_best.txt"
# MAGIC
# MAGIC     headers = {
# MAGIC     "Authorization": f"Bearer {token}",
# MAGIC     "Content-Type": "application/octet-stream",
# MAGIC     }
# MAGIC
# MAGIC     resp = requests.put(
# MAGIC     f"{host}/api/2.0/fs/files{target}?overwrite=true",
# MAGIC     headers=headers,
# MAGIC     data=buf          
# MAGIC     )
# MAGIC     return 'eda_upload done {target}'
# MAGIC    
# MAGIC ############### HYPER PARAM CALUCLATION ####################
# MAGIC
# MAGIC def ask_claude_for_hyperparams(result):
# MAGIC     eda_prompt_stats = result
# MAGIC
# MAGIC     system_prompt_hyperparams = """
# MAGIC     You are an expert machine learning engineer.
# MAGIC
# MAGIC     Your task:
# MAGIC     - You receive, in the user message, EDA statistics for a tabular dataset as a JSON object.
# MAGIC     - Based on those statistics, you must design hyperparameter search spaces for multiple scikit-learn models which are not compute expensive. The hyperparamters that are genereated should be light weight.
# MAGIC     - You must output ONLY a compact JSON object (no explanations, no markdown, no comments).
# MAGIC
# MAGIC     Input format (from user):
# MAGIC     - The user will provide:
# MAGIC     - "feature_list": string with comma-separated feature names (e.g. "pH,density,sulphates").
# MAGIC     - "target_col": name of the target column.
# MAGIC     - "row_count": integer number of rows in the dataset (may be sample or full).
# MAGIC     - "columns": a JSON object where each key is a column name and each value is an object with:
# MAGIC         - "dtype": data type string (for example "double", "bigint", "string").
# MAGIC         - "null_pct": fraction of missing values in [0, 1], or null if unknown.
# MAGIC         - "unique_values": list of unique values for that column if available, else null.
# MAGIC         - "min": numeric minimum for the column if available, else null.
# MAGIC         - "max": numeric maximum for the column if available, else null.
# MAGIC         - "mean": numeric mean for the column if available, else null.
# MAGIC
# MAGIC     Your output:
# MAGIC     - Return ONLY a JSON object (valid JSON, no trailing commas, no extra text) describing hyperparameter grids for several scikit-learn models like logistic regression, naive bayes,decision trees.
# MAGIC     - The keys of the top-level JSON should be model identifiers (for example "log_reg", "nb",dt).
# MAGIC     - Each model value must be an object whose keys are scikit-learn parameter names (prefixed with the Pipeline step name, e.g. "clf__C"), and whose values are arrays of candidate values.
# MAGIC     - Example of the expected shape (the exact values must be adapted to the actual statistics):
# MAGIC     {
# MAGIC         "log_reg": {
# MAGIC         "clf__C": [0.1, 1.0, 10.0],
# MAGIC         "clf__penalty": ["l2"],
# MAGIC         "clf__solver": ["lbfgs"]
# MAGIC         },
# MAGIC         "nb": {
# MAGIC         "clf__var_smoothing": [1e-9, 1e-8, 1e-7]
# MAGIC         }
# MAGIC         "dt": {
# MAGIC         "clf__max_depth": [3, 5, 7],
# MAGIC         "clf__min_samples_split": [2, 5, 7],
# MAGIC         "clf__min_samples_leaf": [1, 2, 5]
# MAGIC         }
# MAGIC     }
# MAGIC
# MAGIC     How to choose hyperparameters:
# MAGIC     - Use "row_count" to decide model complexity and grid size:
# MAGIC     - For small datasets, keep trees shallow and limit n_estimators.
# MAGIC     - For larger datasets, you may allow deeper trees and more estimators.
# MAGIC     - Use "columns" statistics to reason about:
# MAGIC     - Number of features (from feature_list) and their dtypes.
# MAGIC     - Presence of missing values (null_pct).
# MAGIC     - Target cardinality and class balance (if unique_values or numeric stats on the target are available).
# MAGIC     - Prefer reasonable, compact grids suited for:
# MAGIC     - Binary vs multiclass classification (based on target_col info).
# MAGIC     - Numeric vs categorical features (if dtype and unique_values indicate many categories).
# MAGIC
# MAGIC     Constraints:
# MAGIC     - Output MUST be valid JSON, with:
# MAGIC     - Double quotes around all keys and string values.
# MAGIC     - null instead of Python None.
# MAGIC     - No comments, no markdown, no explanatory text.
# MAGIC     - If some statistics are missing (e.g. null min/max/mean), still choose sensible default ranges based on typical practice, not on assumptions about the missing values """
# MAGIC
# MAGIC     eda_prompt_hyperparams = f"""
# MAGIC     {{
# MAGIC         "feature_list": "{feature_list}",
# MAGIC         "target_col": "{target_col}",
# MAGIC         "row_count": 4527,
# MAGIC         "columns": {json.dumps(eda_prompt_stats)}
# MAGIC     }}
# MAGIC     """
# MAGIC     ########### HYPER PARAM GENERATOR USING AGENT ########################
# MAGIC
# MAGIC     # Use secrets or environment variables in real code
# MAGIC     DATABRICKS_MODEL_URL = "https://cust-e2-us-west-2.cloud.databricks.com/serving-endpoints/databricks-gpt-5-1/invocations"
# MAGIC
# MAGIC     headers = {
# MAGIC         "Authorization": f"Bearer ",
# MAGIC         "Content-Type": "application/json",
# MAGIC     }
# MAGIC     payload = {
# MAGIC         "messages": [
# MAGIC             {
# MAGIC                 "role": "system",
# MAGIC                 "content": system_prompt_hyperparams
# MAGIC             },
# MAGIC             {
# MAGIC                 "role": "user",
# MAGIC                 "content": eda_prompt_hyperparams
# MAGIC             }
# MAGIC         ]
# MAGIC     }
# MAGIC     resp = requests.post(DATABRICKS_MODEL_URL, headers=headers, json=payload)
# MAGIC
# MAGIC     resp.raise_for_status()
# MAGIC     content = resp.json()["choices"][0]["message"]["content"]
# MAGIC     grids = json.loads(content)
# MAGIC     return json.dumps(grids)
# MAGIC
# MAGIC ########################### MODEL TRAIN #########################
# MAGIC
# MAGIC def model_train(param_grids, pdf,target,feature_list,model_path):
# MAGIC     X = pdf[feature_list]
# MAGIC     y = pdf[target]
# MAGIC
# MAGIC     pipelines = {
# MAGIC     "log_reg": Pipeline([
# MAGIC         ("scaler", StandardScaler()),
# MAGIC         ("clf", LogisticRegression())
# MAGIC     ]),
# MAGIC     "dt":Pipeline([
# MAGIC         ("scaler", StandardScaler()),
# MAGIC         ("clf", DecisionTreeClassifier())
# MAGIC     ]),
# MAGIC     "nb": Pipeline([
# MAGIC         ("scaler", StandardScaler()),
# MAGIC         ("clf", GaussianNB())
# MAGIC     ])
# MAGIC     }
# MAGIC     results = []
# MAGIC
# MAGIC     best_overall_score = -1
# MAGIC     best_overall_model = None
# MAGIC     best_overall_name = None
# MAGIC
# MAGIC     for name in pipelines:
# MAGIC         grid = GridSearchCV(
# MAGIC             estimator=pipelines[name],
# MAGIC             param_grid=param_grids[name],
# MAGIC             scoring="accuracy",
# MAGIC             cv = 3,
# MAGIC             n_jobs=1
# MAGIC         )
# MAGIC         
# MAGIC         grid.fit(X, y)
# MAGIC         
# MAGIC         results.append({
# MAGIC             "model": name,
# MAGIC             "best_score": grid.best_score_,
# MAGIC             "best_params": grid.best_params_
# MAGIC         })
# MAGIC         
# MAGIC         if grid.best_score_ > best_overall_score:
# MAGIC             best_overall_score = grid.best_score_
# MAGIC             best_overall_model = grid.best_estimator_
# MAGIC             best_overall_name = name
# MAGIC
# MAGIC     ########## MODEL JOBLIB WRITE ################
# MAGIC
# MAGIC     buf = io.BytesIO()
# MAGIC     dump(best_overall_model, buf)
# MAGIC     buf.seek(0)
# MAGIC     host = "https://cust-e2-us-west-2.cloud.databricks.com"
# MAGIC     token = ""
# MAGIC     model_path = build_model_path(source_table, feature_list, target_col)
# MAGIC     model_file = model_path
# MAGIC
# MAGIC
# MAGIC     headers = {
# MAGIC     "Authorization": f"Bearer {token}",
# MAGIC     "Content-Type": "application/octet-stream",
# MAGIC     }
# MAGIC
# MAGIC     resp = requests.put(
# MAGIC     f"{host}/api/2.0/fs/files{model_file}?overwrite=true",
# MAGIC     headers=headers,
# MAGIC     data=buf          
# MAGIC     )
# MAGIC
# MAGIC     resp.raise_for_status()
# MAGIC     
# MAGIC     return best_overall_model
# MAGIC
# MAGIC
# MAGIC def ai_ml_classify_implementation(source_table, feature_list, target_col, train):
# MAGIC
# MAGIC     server_hostname = "cust-e2-us-west-2.cloud.databricks.com"
# MAGIC     http_path = "/sql/1.0/warehouses/85f0f50fcfcd8a88"
# MAGIC     access_token = ""
# MAGIC
# MAGIC     should_train = False
# MAGIC     model_path = build_model_path(source_table, feature_list, target_col)
# MAGIC     model_file = model_path
# MAGIC
# MAGIC
# MAGIC     headers = {
# MAGIC     "Authorization": f"Bearer {access_token}",
# MAGIC     "Content-Type": "application/octet-stream",
# MAGIC     }
# MAGIC
# MAGIC     resp = requests.get(
# MAGIC     f"https://{server_hostname}/api/2.0/fs/files{model_file}",
# MAGIC     headers=headers          
# MAGIC     )
# MAGIC
# MAGIC     #resp.raise_for_status()
# MAGIC
# MAGIC     if str(resp.status_code) != '200':
# MAGIC         should_train = True
# MAGIC     elif train is True:
# MAGIC         should_train = True
# MAGIC     else:
# MAGIC         should_train = False
# MAGIC
# MAGIC     #should_train = False
# MAGIC     
# MAGIC     if should_train == True:
# MAGIC         pdf = getPandasData(server_hostname, http_path, access_token, source_table)
# MAGIC         rows_json = pdf.to_json(orient="records")
# MAGIC
# MAGIC         eda_stats = transform(rows_json, feature_list, target_col)
# MAGIC         eda_stats = build_column_stats(eda_stats)
# MAGIC         hyperparams = ask_gpt_for_eda(eda_stats)
# MAGIC
# MAGIC         #resp.raise_for_status()
# MAGIC         hyper_params = ask_claude_for_hyperparams(eda_stats)
# MAGIC         hyper_params_json = json.loads(hyper_params)
# MAGIC         model_path = build_model_path(source_table, feature_list, target_col)
# MAGIC         model_train_best_model = model_train(hyper_params_json, pdf, target_col,feature_list,model_path)
# MAGIC         pdf_final = pd.DataFrame([feature_schema])[feature_list]
# MAGIC
# MAGIC         
# MAGIC         prediction = model_train_best_model.predict(pdf_final)[0]
# MAGIC         return (prediction)
# MAGIC
# MAGIC     else:
# MAGIC       model_file = "/Volumes/jahnavi/def/ai_ml_classify/model_best.joblib"
# MAGIC
# MAGIC       url = f"https://cust-e2-us-west-2.cloud.databricks.com/api/2.0/fs/files{model_file}"
# MAGIC       headers = {
# MAGIC           "Authorization": "Bearer ",
# MAGIC           "Content-Type": "application/octet-stream",
# MAGIC       }
# MAGIC
# MAGIC       resp = requests.get(url, headers=headers)
# MAGIC       resp.raise_for_status()
# MAGIC
# MAGIC       buf = io.BytesIO(resp.content)
# MAGIC       buf.seek(0)
# MAGIC       model = load(buf)
# MAGIC
# MAGIC       pdf = pd.DataFrame([feature_schema])[feature_list]
# MAGIC       prediction = model.predict(pdf)[0]
# MAGIC       return (prediction)
# MAGIC
# MAGIC return ai_ml_classify_implementation(source_table , feature_list, target_col, train)     
# MAGIC $$;
# MAGIC