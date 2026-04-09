import json
import pandas as pd


def change_jsonl_to_csv(input_file, output_file, prompt_column="prompt", response_column="response"):
    prompts = []
    responses = []
    with open(input_file, 'r') as json_file:
        for data in json_file:
            prompts.append(json.loads(data)[0]['messages'][0]['content'])
            responses.append(json.loads(data)[1]['choices'][0]['message']['content'])

    df = pd.DataFrame({prompt_column: prompts, response_column: responses})
    df.to_csv(output_file, index=False)
    return df


def merge_gt_and_gen_result(df_gt, df_gen):
    results = []
    for idx, row in df_gen.iterrows():
        with_sql_gt = df_gt.loc[df_gt['without_sql'] == row['without_sql']] 
        gt_sql = with_sql_gt['sql'].values[0]
        gen_sql = row['gen_sql']
        results.append((with_sql_gt['ddl'].values[0], with_sql_gt['request'].values[0], gt_sql, gen_sql))
    df_result = pd.DataFrame(results, columns=["ddl", "request", "gt_sql", "gen_sql"])
    return df_result

def make_evaluation_requests(df, filename, model="gpt-4-1106-preview"):
    prompts = []
    for idx, row in df.iterrows():
        prompts.append(f"""Based on provided ddl, request, gen_sql, ground_truth_sql if gen_sql eqauls to ground_truth_sql, output "yes" else "no"
DDL:
{row['ddl']}
Request:
{row['request']}
ground_truth_sql:
{row['gt_sql']}
gen_sql:
{row['gen_sql']}

Answer:""")

    jobs = [{"model": model, "messages": [{"role": "system", "content": prompt}]} for prompt in prompts]
    with open(filename, "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")

def make_prompt(ddl, request, sql=""):
    prompt = f"""당신은 SQL을 생성하는 SQL 봇입니다. DDL과 요청사항을 바탕으로 적절한 SQL 쿼리를 생성하세요.

DDL:
{ddl}

요청사항:
{request}

SQL:
{sql}"""
    return prompt


if __name__ == '__main__':
    df = pd.read_csv('./nl2sql_validation.csv')
    df.sample(100).to_csv('nl2sql_validation_sample.csv', index=False)