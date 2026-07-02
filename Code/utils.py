"""General loading and export utilities."""

# ----------------------------------
#       Importing libraries
# ----------------------------------

import os, re, json
import fitz
import tiktoken
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import textract
except ImportError:
    textract = None



# ----------------------------------
#       Loading the data
# ----------------------------------

def text_cleaner(text):
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    text = re.sub(r'\t{2,}', '\t', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()



def load_documents(
        file_path: str | Path, 
        file_name_header: Optional[str] = 'file_name',
        file_content_header: Optional[str] = 'file_content',
        verbose: Optional[bool] = False):
    """Load documents from a folder into a dict of text lists."""
    
    file_path = Path(file_path).resolve()

    data = {
        file_name_header: [],
        file_content_header: []
    }
    
    for dirpath, _, filenames in os.walk(file_path):
        for file in filenames:
            data[file_name_header].append(os.path.splitext(file)[0])
            file_path = Path(dirpath)/file
            if file.endswith('.pdf'):
                doc = fitz.open(file_path)
                text = ''
                for page in doc:
                    text += page.get_text()
                    text = text_cleaner(text)
                data[file_content_header].append(text)
            else:
                try:
                    if textract is None:
                        raise ImportError("textract is not installed")
                    text = textract.process(str(file_path)).decode('utf-8')
                    text = text_cleaner(text)
                    data[file_content_header].append(text)
                except Exception as e:
                    print(f"""Error {e} for file {file} in {file_path};
                          Check if the file is corrupted or not supported, 
                          then check textract dependencies on your environment.""")
            if verbose:
                print(f'File name {file} is loaded successfully!')
    return data
                


def token_count(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoding.encode(text))
    return token_count



# ----------------------------------
#       Saving the results
# ----------------------------------

def csv_output(data: dict,
               csv_file_name: Optional[str] = 'evaluation_results.csv',
               file_name_header: Optional[str] = 'file_name',
               file_content_header: Optional[str] = 'file_content'):
    
    df = pd.DataFrame.from_dict(data)

    # calculating the sum and mean of the evaluation scores:
    token_outputs = ['input_token_header',
                    'prompt_tokens', 'total_tokens', 'processing_time']
    non_numeric_cols = [file_name_header, file_content_header, 'title']
    non_calculatable = token_outputs + non_numeric_cols
    calculation_cols = [col for col in list(df.columns) if col not in non_calculatable]
    
    df['scores_sum'] = df.loc[:, calculation_cols].sum(axis=1)
    df['scores_mean'] = df.loc[:, calculation_cols].mean(axis=1)
    df['scores_mean_rounded'] = round(df['scores_mean']).astype(int)

    # putting file_name_header, file_content_header at columns 1 and 2:
    first_cols = [file_name_header, 'title', file_content_header]
    cols = first_cols + [col for col in list(df.columns) if col not in first_cols]
    df = df[cols]

    df = df.sort_values(by='scores_mean')
    df.to_csv(csv_file_name, encoding='utf-8', index=False)
    print(f"The evaluation results are saved and accessible from {csv_file_name}.")



def json_to_excel(json_path: str, output_path: str = None):
    """Convert a JSON result file to Excel."""

    json_path = Path(json_path).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"[ERROR] JSON file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        if any(isinstance(v, list) for v in data.values()):
            data = pd.json_normalize(data, sep='.')
        else:
            data = [data]
    df = pd.json_normalize(data, sep='.')

    # Calculating summary statistics for numeric criteria
    numeric_cols = []
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and col not in [
            'input_token_count', 'prompt_tokens', 'total_tokens', 'processing_time'
        ]:
            numeric_cols.append(col)
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            df[col] = df[col].apply(lambda x: x*10)
            numeric_cols.append(col)
    
    if numeric_cols:
        df['scores_sum'] = df[numeric_cols].sum(axis=1, skipna=True)
        df['scores_mean'] = df[numeric_cols].mean(axis=1, skipna=True)
        df['scores_mean_rounded'] = df['scores_mean'].round().astype('Int64')

    if output_path is None:
        output_path = json_path.with_suffix('.xlsx')
    else:
        output_path = Path(output_path).resolve()
        if output_path.is_dir():
            output_path = output_path / f"{json_path.stem}.xlsx"

    df.to_excel(output_path, index=False, engine='openpyxl')

    print(f"Excel file created: {output_path}")
    return output_path
