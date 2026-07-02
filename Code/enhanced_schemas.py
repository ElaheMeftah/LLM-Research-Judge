"""Main schema and output helpers for abstract evaluation."""

from typing import Optional, Union, Literal
from pydantic import BaseModel, Field, create_model
from langchain.output_parsers import PydanticOutputParser
import pandas as pd
import json
from pathlib import Path
import openpyxl


def abstract_criteria_definer(user_criteria: dict = None,
                              criteria_to_keep: list = None) -> PydanticOutputParser:
    """Build the default abstract evaluation parser."""
    
    # LLM-GENERATED EVALUATION CRITERIA
    llm_evaluation_fields = {
        # Content Quality (1-10 scale)
        'title_quality': (Optional[int], Field(None, ge=1, le=10,
            description="Quality and appropriateness of the title")),
        'relevance': (Optional[int], Field(None, ge=1, le=10,
            description="Relevance to the field and conference theme(s)")),
        'about_congress_special_theme': (Optional[int], Field(None, ge=1, le=10,
            description="Addresses the special theme(s) of the congress")),
        'innovation': (Optional[int], Field(None, ge=1, le=10,
            description="Novelty and innovative aspects of the research")),
        'global_impact': (Optional[int], Field(None, ge=1, le=10,
            description="Addressing critical gaps in global/public health as stated by WHO")),
        'equity': (Optional[int], Field(None, ge=1, le=10,
            description="Addressing social determinants of infectious disease burden")),
        'scientifically_sound': (Optional[int], Field(None, ge=1, le=10,
            description="Scientific rigor and validity of the research")),
        'ethically_sound': (Optional[int], Field(None, ge=1, le=10,
            description="Ethical considerations and compliance")),
        'multidisciplinary': (Optional[int], Field(None, ge=1, le=10,
            description="Interdisciplinary approach and collaboration")),
        'multicenter': (Optional[bool], Field(None,
            description="Based on multiple centers/institutions according to methods")),
        
        # Writing Quality (1-10 scale)
        'context_and_background': (Optional[int], Field(None, ge=1, le=10,
            description="Summarizes what is already known and defines key terms for topic understanding")),
        'knowledge_gap_identification': (Optional[int], Field(None, ge=1, le=10,
            description="Clearly states what remains unknown and why filling the gap is important")),
        'rationale_and_significance': (Optional[int], Field(None, ge=1, le=10,
            description="Explains study novelty and potential impact on science, clinical care, or policy")),
        'logical_flow_and_structure': (Optional[int], Field(None, ge=1, le=10,
            description="Each sentence and paragraph leads logically to the next")),
        'tone_and_readability': (Optional[int], Field(None, ge=1, le=10,
            description="Precise, engaging language that avoids unnecessary jargon")),
        'language_and_grammar': (Optional[int], Field(None, ge=1, le=10,
            description="Correctness, clarity, fluency and adherence to scientific writing standards")),
        'human_generated': (Optional[int], Field(None, ge=1, le=10,
            description="How human vs AI-generated the abstract appears (1=fully AI, 10=fully human)")),
        'length': (Optional[int], Field(None, ge=1, le=10,
            description="Appropriateness of abstract length (standard: 150-300 words)")),
        'keyword_quality': (Optional[int], Field(None, ge=1, le=10,
            description="Quality and relevance of provided keywords")),
        
        # Methodological Quality (1-10 scale)
        'appropriate_study_design': (Optional[int], Field(None, ge=1, le=10,
            description="Study design is suitable for the research question")),
        'methodological_transparency': (Optional[int], Field(None, ge=1, le=10,
            description="Methods are clearly described and transparent")),
        'proper_sampling': (Optional[int], Field(None, ge=1, le=10,
            description="Sample size is properly calculated and described")),
        'controls_comparators': (Optional[int], Field(None, ge=1, le=10,
            description="Appropriate controls or comparators are used when applicable")),
        'statistical_analysis': (Optional[int], Field(None, ge=1, le=10,
            description="Statistical methods are valid and properly applied")),
        'bias_and_confounding_control': (Optional[int], Field(None, ge=1, le=10,
            description="Adequate control of bias and confounding factors")),
        'outcome_reporting': (Optional[int], Field(None, ge=1, le=10,
            description="Outcomes quantified with proper reporting, not just p-values")),
        'result_plausibility': (Optional[int], Field(None, ge=1, le=10,
            description="Results are plausible and reliable")),
        'missing_data_management': (Optional[int], Field(None, ge=1, le=10,
            description="Appropriate handling of missing data")),
        'results_and_text_concordance': (Optional[int], Field(None, ge=1, le=10,
            description="Consistency between reported results and narrative text")),
        'affiliations_and_text_concordance': (Optional[bool], Field(None,
            description="False if the primary institution stated in the abstract is not represented in any of the authors' affiliations. Consider it True for multi-center studies that list multiple countries, as long as the affiliations cover at least one of them.")),
        
        # Structure and Presentation (1-10 scale)
        # 'structured_format': (Optional[int], Field(None, ge=1, le=10,
        #     description="Logical abstract structure (Background-Methods-Results-Conclusion)")),
        'objectives': (Optional[int], Field(None, ge=1, le=10,
            description="Research aims and objectives are clearly stated")),
        'conclusion': (Optional[int], Field(None, ge=1, le=10,
            description="Conclusions are supported by the presented results")),
        'translational_potential_explanation': (Optional[int], Field(None, ge=1, le=10,
            description="Clinical or practical implications are discussed")),
        'future_directions': (Optional[int], Field(None, ge=1, le=10,
            description="Future research directions are appropriately discussed")),
        
        # Overall Assessment
        'overall_evaluation': (Optional[int], Field(None, ge=1, le=10,
            description="Overall quality assessment of the abstract")),
        'comments_for_improvement': (Optional[str], Field(None,
            description="Concise, critical, specific, and actionable suggestions for improvement, written as bullet points")),
        'accept_or_reject': (Literal['Rejected', 'Accepted_Poster', 'Accepted_Oral'], Field(None,
            description="Final recommendation: 'Rejected', 'Accepted_Poster', or 'Accepted_Oral'"))
    }
    
    
    # Filter criteria if specified
    if criteria_to_keep:
        selected_criteria = {
            key: val for key, val in llm_evaluation_fields.items() 
            if key in criteria_to_keep
        }
    else:
        selected_criteria = llm_evaluation_fields
    
    # Add user-defined criteria if provided
    if user_criteria:
        user_defined_criteria = {}
        for key, value in user_criteria.items():
            if isinstance(value, str):
                # Text criteria
                user_defined_criteria[key] = (Optional[str], Field(None, description=value))
            else:
                # Assume numeric criteria (1-10 scale)
                user_defined_criteria[key] = (Optional[int], Field(None, ge=1, le=10, description=str(value)))
    else:
        user_defined_criteria = {}
    
    # Combine all criteria
    final_criteria = {**selected_criteria, **user_defined_criteria}
    
    # Create the Pydantic model
    pydantic_model = create_model('AbstractEvaluationCriteria', **final_criteria)
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    
    return parser


def create_evaluation_output(evaluation_data: dict, 
                           output_dir: str = ".", 
                           base_filename: str = "abstract_evaluation") -> tuple[str, str]:
    """Save evaluation results as Excel and JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Excel file
    excel_path = output_dir / f"{base_filename}.xlsx"
    df = pd.DataFrame.from_dict(evaluation_data)
    
    # Cols with dual importance as defined based on expert opinion
    COLS_2X = ['about_congress_special_theme', 'innovation', 'global_impact', 
               'equity', 'multicenter', 'knowledge_gap_identification', 
               'rationale_and_significance', 'appropriate_study_design',
               'methodological_transparency', 'proper_sampling', 'controls_comparators', 
               'statistical_analysis', 'bias_and_confounding_control', 'conclusion']

    # Apply 2x weighting only to columns that exist in the dataframe
    existing_2x_cols = [col for col in COLS_2X if col in df.columns]
    if existing_2x_cols:
        df[existing_2x_cols] = df[existing_2x_cols].apply(lambda x: x*2)

    # Calculate summary statistics for numeric criteria
    numeric_cols = []
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and col not in [
            'input_token_count', 'prompt_tokens', 'total_tokens', 'processing_time'
        ]:
            numeric_cols.append(col)
    
    if numeric_cols:
        df['scores_sum'] = df[numeric_cols].sum(axis=1, skipna=True)
        df['scores_mean'] = df[numeric_cols].mean(axis=1, skipna=True)
        df['scores_mean_rounded'] = df['scores_mean'].round().astype('Int64')
    
    # Reorder columns for better readability
    priority_cols = ['abstract_code', 'title', 'accept_or_reject', 'overall_evaluation']
    other_cols = [col for col in df.columns if col not in priority_cols]
    ordered_cols = [col for col in priority_cols if col in df.columns] + other_cols
    df = df[ordered_cols]
    
    # Save Excel with multiple sheets
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Evaluation_Results', index=False)
        
        # Create summary sheet if we have numeric data
        if numeric_cols:
            summary_data = {
                'Criteria': numeric_cols,
                'Mean_Score': [df[col].mean() for col in numeric_cols],
                'Std_Score': [df[col].std() for col in numeric_cols],
                'Min_Score': [df[col].min() for col in numeric_cols],
                'Max_Score': [df[col].max() for col in numeric_cols]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    # Create JSON file
    json_path = output_dir / f"{base_filename}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
    
    return str(excel_path), str(json_path)


def prepare_abstracts_for_evaluation(parsed_abstracts: list, 
                                    file_content_header: str = 'file_content',
                                    file_name_header: str = 'file_name') -> dict:
    """Convert parsed abstracts into prompt-ready text."""
    prepared_data = {
        file_name_header: [],
        file_content_header: []
    }
    
    for abstract in parsed_abstracts:
        # Use abstract_code as filename
        abstract_id = abstract.get('abstract_code', 'Unknown')
        prepared_data[file_name_header].append(abstract_id)
        
        # Create formatted content for LLM evaluation
        content_parts = []

        # Add title
        if abstract.get('Title'):
            content_parts.append(f"Title: {abstract['Title']}")
        
        # Add authors
        if abstract.get('Authors'):
            content_parts.append(f"Authors: {abstract['Authors']}")
        
        # Add affiliations
        if abstract.get('Affiliations'):
            content_parts.append(f"Affiliations: {abstract['Affiliations']}")
        
        # Add main content sections
        sections = ['Introduction', 'Methods', 'Results', 'Conclusions']
        for section in sections:
            if abstract.get(section):
                content_parts.append(f"{section}: {abstract[section]}")
        
        # Add keywords
        if abstract.get('Keywords'):
            content_parts.append(f"Keywords: {abstract['Keywords']}")
        
        # Join all parts
        full_content = '\n\n'.join(content_parts)
        prepared_data[file_content_header].append(full_content)
    
    return prepared_data


# Example usage function
def evaluate_abstracts_complete_workflow(parsed_abstracts_file: str,
                                       output_dir: str = "evaluation_results",
                                       user_criteria: dict = None,
                                       criteria_to_keep: list = None) -> tuple[str, str]:
    """Prepare a sample evaluation output without calling an LLM."""
    # Load parsed abstracts
    with open(parsed_abstracts_file, 'r', encoding='utf-8') as f:
        parsed_abstracts = json.load(f)
    
    # Prepare data for evaluation
    evaluation_data = prepare_abstracts_for_evaluation(parsed_abstracts)
    
    # Get the parser with enhanced schema
    parser = abstract_criteria_definer(user_criteria=user_criteria, 
                                     criteria_to_keep=criteria_to_keep)
    
    print(f"Schema created with {len(parser.pydantic_object.__fields__)} evaluation criteria")
    print(f"Ready to evaluate {len(parsed_abstracts)} abstracts")
    
    # Create sample output structure (replace with actual LLM evaluation)
    sample_result = {}
    for key in evaluation_data.keys():
        sample_result[key] = evaluation_data[key]
    
    # Add sample evaluation scores for demonstration
    for abstract in parsed_abstracts:
        for field_name in parser.pydantic_object.__fields__.keys():
            if field_name not in sample_result:
                sample_result[field_name] = []
            # Add placeholder values
            if field_name in ['title', 'authors', 'affiliations', 'introduction', 'methods', 'results', 'conclusions', 'keywords']:
                sample_result[field_name].append(abstract.get(field_name.capitalize()))
            else:
                sample_result[field_name].append(None)  # LLM will fill these
    
    # Create output files
    excel_path, json_path = create_evaluation_output(sample_result, output_dir)
    
    return excel_path, json_path
