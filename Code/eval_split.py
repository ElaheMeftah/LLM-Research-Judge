"""Split-schema abstract evaluation workflows."""

import json
from pathlib import Path

from .flexible_schemas import (
    abstract_criteria_split_part_a_numeric,
    abstract_criteria_split_part_b_numeric,
    abstract_criteria_split_part_a_categorical,
    abstract_criteria_split_part_b_categorical,
    merge_split_evaluation_results,
    convert_categorical_to_numeric,
)
from .enhanced_schemas import (
    prepare_abstracts_for_evaluation,
    create_evaluation_output
)
from .generator import openai_response_generator


def create_evaluation_prompt_part_a():
    """Return the content and writing prompt."""
    prompt = """
    You are an expert reviewer for a medical conference specializing in infectious diseases and microbiology. 
    Your task is to critically evaluate the following research abstract focusing on CONTENT and WRITING QUALITY.
    Be objective and strict in your evaluation.
    
    This is Part A of a two-part evaluation. Focus on:
    - Overall content quality and innovation
    - Writing style, clarity, and structure
    - Background and rationale presentation
    - Title and keyword quality
    
    Please evaluate the abstract thoroughly and provide scores for each criterion on a scale of 1-10 
    (where 1 is very poor and 10 is excellent) unless otherwise specified.
    
    For boolean fields, please use true or false.
    
    Abstract to evaluate:
    {intro_text}
    
    Please provide your evaluation in the exact JSON format specified below:
    {format_instructions}

    """
    
    return prompt


def create_evaluation_prompt_part_b():
    """Return the methods and overall assessment prompt."""
    prompt = """
    You are an expert reviewer for a medical conference specializing in infectious diseases and microbiology. 
    Your task is to evaluate the following research abstract focusing on METHODOLOGICAL QUALITY and OVERALL ASSESSMENT.
    
    This is Part B of a two-part evaluation. Focus on:
    - Methodological rigor and transparency
    - Study design appropriateness
    - Statistical analysis quality
    - Results presentation and conclusions
    - Overall assessment and recommendations
    
    Please evaluate the abstract thoroughly and provide scores for each criterion on a scale of 1-10 
    (where 1 is very poor and 10 is excellent) unless otherwise specified.
    
    For the overall assessment, provide specific constructive feedback and a final recommendation.
    
    Abstract to evaluate:
    {intro_text}
    
    Please provide your evaluation in the exact JSON format specified below:
    {format_instructions}
    
    Important evaluation guidelines for Part B:
    - Assess the appropriateness and rigor of the methodology
    - Evaluate the quality of statistical analysis and results reporting
    - Consider the logical structure and presentation format
    - Provide constructive feedback for improvement
    - Make a final recommendation: "Rejected", "Accepted_Poster", or "Accepted_Oral"
    """
    
    return prompt


def create_evaluation_prompt_part_a_categorical():
    """Return the categorical content and writing prompt."""
    prompt = """
    You are an expert reviewer for a medical conference specializing in infectious diseases and microbiology. 
    Your task is to evaluate the following research abstract focusing on CONTENT and WRITING QUALITY.
    
    This is Part A of a two-part evaluation. Focus on:
    - Overall content quality and innovation
    - Writing style, clarity, and structure
    - Background and rationale presentation
    - Title and keyword quality
    
    Please evaluate the abstract thoroughly using a categorical scale (low/medium/high) 
    where:
    - low: Poor quality, significant issues, needs major improvement
    - medium: Acceptable quality, some issues, needs minor to moderate improvement
    - high: Excellent quality, minimal issues, publication-ready
    
    For boolean fields, please use true or false.
    
    Abstract to evaluate:
    {intro_text}
    
    Please provide your evaluation in the exact JSON format specified below:
    {format_instructions}
    
    """
    
    return prompt


def create_evaluation_prompt_part_b_categorical():
    """Return the categorical methods and assessment prompt."""
    prompt = """
    You are an expert reviewer for a medical conference specializing in infectious diseases and microbiology. 
    Your task is to evaluate the following research abstract focusing on METHODOLOGICAL QUALITY and OVERALL ASSESSMENT.
    
    This is Part B of a two-part evaluation. Focus on:
    - Methodological rigor and transparency
    - Study design appropriateness
    - Statistical analysis quality
    - Results presentation and conclusions
    - Overall assessment and recommendations
    
    Please evaluate the abstract thoroughly using a categorical scale (low/medium/high) 
    where:
    - low: Poor quality, significant issues, needs major improvement
    - medium: Acceptable quality, some issues, needs minor to moderate improvement
    - high: Excellent quality, minimal issues, publication ready
    
    For the overall assessment, provide specific constructive feedback and a final recommendation.
    
    Abstract to evaluate:
    {intro_text}
    
    Please provide your evaluation in the exact JSON format specified below:
    {format_instructions}
    
    Important evaluation guidelines for Part B:
    - Assess the appropriateness and rigor of the methodology
    - Evaluate the quality of statistical analysis and results reporting
    - Consider the logical structure and presentation format
    - Provide constructive feedback for improvement
    - Make a final recommendation: "Rejected", "Accepted_Poster", or "Accepted_Oral"

    """
    
    return prompt


def run_split_evaluation_numeric(parsed_abstracts_file: str,
                               output_dir: str = "evaluation_results",
                               model: str = "gpt-4.1-mini",
                               max_abstracts: int = None) -> tuple[str, str]:
    """Run split evaluation with 1-10 scores."""
    
    print("Starting split evaluation workflow (numeric ratings)...")
    print(f"Using model: {model}")
    
    # Load parsed abstracts
    with open(parsed_abstracts_file, 'r', encoding='utf-8') as f:
        parsed_abstracts = json.load(f)
    
    if max_abstracts:
        parsed_abstracts = parsed_abstracts[:max_abstracts]
        print(f"Limited to first {max_abstracts} abstracts for testing")
    
    print(f"Loaded {len(parsed_abstracts)} abstracts for evaluation")
    
    # Prepare data for evaluation
    evaluation_data = prepare_abstracts_for_evaluation(parsed_abstracts)
    
    # Create parsers for both parts
    parser_a = abstract_criteria_split_part_a_numeric()
    parser_b = abstract_criteria_split_part_b_numeric()
    
    print(f"Part A schema: {len(parser_a.pydantic_object.__fields__)} criteria")
    print(f"Part B schema: {len(parser_b.pydantic_object.__fields__)} criteria")
    
    # Create prompts
    prompt_a = create_evaluation_prompt_part_a()
    prompt_b = create_evaluation_prompt_part_b()
    
    print("Starting Part A evaluation (Content + Writing Quality)...")
    
    # Run Part A evaluation
    results_a = openai_response_generator(
        data=evaluation_data,
        prompt=prompt_a,
        parser=parser_a,
        verbose=True,
        model=model,
        max_tokens=3000
    )
    
    print("Part A evaluation completed!")
    print("Starting Part B evaluation (Methodology + Overall Assessment)...")
    
    # Run Part B evaluation
    results_b = openai_response_generator(
        data=evaluation_data,
        prompt=prompt_b,
        parser=parser_b,
        verbose=True,
        model=model,
        max_tokens=3000
    )
    
    print("Part B evaluation completed!")
    print("Merging results...")
    
    # Merge the results
    merged_results = merge_split_evaluation_results(results_a, results_b)
    
    # Create output files
    excel_path, json_path = create_evaluation_output(
        merged_results, 
        output_dir=output_dir,
        base_filename=f"split_evaluation_numeric_{model.replace('-', '_')}"
    )
    
    print(f"Results saved to:")
    print(f"  Excel: {excel_path}")
    print(f"  JSON: {json_path}")
    
    return excel_path, json_path


def run_split_evaluation_categorical(parsed_abstracts_file: str,
                                   output_dir: str = "evaluation_results",
                                   model: str = "gpt-4.1-mini",
                                   max_abstracts: int = None,
                                   convert_to_numeric: bool = False) -> tuple[str, str]:
    """Run split evaluation with low, medium, or high ratings."""
    
    print("Starting split evaluation workflow (categorical ratings)...")
    print(f"Using model: {model}")
    
    # Load parsed abstracts
    with open(parsed_abstracts_file, 'r', encoding='utf-8') as f:
        parsed_abstracts = json.load(f)
    
    if max_abstracts:
        parsed_abstracts = parsed_abstracts[:max_abstracts]
        print(f"Limited to first {max_abstracts} abstracts for testing")
    
    print(f"Loaded {len(parsed_abstracts)} abstracts for evaluation")
    
    # Prepare data for evaluation
    evaluation_data = prepare_abstracts_for_evaluation(parsed_abstracts)
    
    # Create parsers for both parts
    parser_a = abstract_criteria_split_part_a_categorical()
    parser_b = abstract_criteria_split_part_b_categorical()
    
    print(f"Part A schema: {len(parser_a.pydantic_object.__fields__)} criteria")
    print(f"Part B schema: {len(parser_b.pydantic_object.__fields__)} criteria")
    
    # Create prompts
    prompt_a = create_evaluation_prompt_part_a_categorical()
    prompt_b = create_evaluation_prompt_part_b_categorical()
    
    print("Starting Part A evaluation (Content + Writing Quality)...")
    
    # Run Part A evaluation
    results_a = openai_response_generator(
        data=evaluation_data,
        prompt=prompt_a,
        parser=parser_a,
        verbose=True,
        model=model,
        max_tokens=3000
    )
    
    print("Part A evaluation completed!")
    print("Starting Part B evaluation (Methodology + Overall Assessment)...")
    
    # Run Part B evaluation
    results_b = openai_response_generator(
        data=evaluation_data,
        prompt=prompt_b,
        parser=parser_b,
        verbose=True,
        model=model,
        max_tokens=3000
    )
    
    print("Part B evaluation completed!")
    print("Merging results...")
    
    # Merge the results
    merged_results = merge_split_evaluation_results(results_a, results_b)
    
    # Convert to numeric if requested
    if convert_to_numeric:
        print("Converting categorical ratings to numeric...")
        merged_results = convert_categorical_to_numeric(merged_results)
        filename_suffix = "categorical_to_numeric"
    else:
        filename_suffix = "categorical"
    
    # Create output files
    excel_path, json_path = create_evaluation_output(
        merged_results, 
        output_dir=output_dir,
        base_filename=f"split_evaluation_{filename_suffix}_{model.replace('-', '_')}"
    )
    
    print(f"Results saved to:")
    print(f"  Excel: {excel_path}")
    print(f"  JSON: {json_path}")
    
    return excel_path, json_path


def compare_evaluation_methods(parsed_abstracts_file: str,
                             output_dir: str = "evaluation_comparison",
                             model: str = "gpt-4.1-mini",
                             max_abstracts: int = 3) -> dict:
    """Run a small comparison of the split evaluation modes."""

    parsed_abstracts_file = Path(parsed_abstracts_file).resolve()
    
    print("Starting evaluation method comparison...")
    print(f"Testing with {max_abstracts} abstracts using model: {model}")
    
    comparison_results = {}
    
    try:
        # Method 1: Split evaluation with numeric ratings
        print("\n1. Testing split evaluation with numeric ratings...")
        excel_path_1, json_path_1 = run_split_evaluation_numeric(
            parsed_abstracts_file, 
            output_dir + "/split_numeric",
            model, 
            max_abstracts
        )
        comparison_results['split_numeric'] = {
            'excel': excel_path_1,
            'json': json_path_1,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"Split numeric evaluation failed: {e}")
        comparison_results['split_numeric'] = {'status': 'failed', 'error': str(e)}
    
    try:
        # Method 2: Split evaluation with categorical ratings
        print("\n2. Testing split evaluation with categorical ratings...")
        excel_path_2, json_path_2 = run_split_evaluation_categorical(
            parsed_abstracts_file, 
            output_dir + "/split_categorical",
            model, 
            max_abstracts,
            convert_to_numeric=False
        )
        comparison_results['split_categorical'] = {
            'excel': excel_path_2,
            'json': json_path_2,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"Split categorical evaluation failed: {e}")
        comparison_results['split_categorical'] = {'status': 'failed', 'error': str(e)}
    
    try:
        # Method 3: Split evaluation with categorical ratings converted to numeric
        print("\n3. Testing split evaluation with categorical ratings (converted to numeric)...")
        excel_path_3, json_path_3 = run_split_evaluation_categorical(
            parsed_abstracts_file, 
            output_dir + "/split_categorical_numeric",
            model, 
            max_abstracts,
            convert_to_numeric=True
        )
        comparison_results['split_categorical_numeric'] = {
            'excel': excel_path_3,
            'json': json_path_3,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"Split categorical-to-numeric evaluation failed: {e}")
        comparison_results['split_categorical_numeric'] = {'status': 'failed', 'error': str(e)}
    
    # Save comparison summary
    summary_path = Path(output_dir) / "comparison_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nComparison completed! Summary saved to: {summary_path}")
    
    # Print summary
    print("\nComparison Summary:")
    for method, result in comparison_results.items():
        status = result['status']
        print(f"  {method}: {status}")
        if status == 'success':
            print(f"    Excel: {result['excel']}")
            print(f"    JSON: {result['json']}")
        else:
            print(f"    Error: {result['error']}")
    
    return comparison_results


def main():
    """Run a small local example."""
    
    # Configuration
    parsed_abstracts_file = Path(__file__).parent.parent / "parsed_abstracts.json"
    output_dir = "split_evaluation_results"
    
    # For testing, limited to first 3 abstracts to save API costs
    max_abstracts_for_testing = 3
    
    # Model selection
    model = "gpt-4.1-mini"
    
    if not parsed_abstracts_file.exists():
        print(f"Error: {parsed_abstracts_file} not found")
        print("Please run the abstract parser first to generate parsed abstracts")
        return
    
    try:
        # Option 1: Run split evaluation with numeric ratings
        print("Option 1: Split evaluation with numeric ratings (1-10 scale)")
        excel_path_1, json_path_1 = run_split_evaluation_numeric(
            parsed_abstracts_file=str(parsed_abstracts_file),
            output_dir=output_dir + "/numeric",
            model=model,
            max_abstracts=max_abstracts_for_testing
        )
        
        # Option 2: Run split evaluation with categorical ratings
        print("\nOption 2: Split evaluation with categorical ratings (low/medium/high)")
        excel_path_2, json_path_2 = run_split_evaluation_categorical(
            parsed_abstracts_file=str(parsed_abstracts_file),
            output_dir=output_dir + "/categorical",
            model=model,
            max_abstracts=max_abstracts_for_testing,
            convert_to_numeric=False
        )
        
        # Option 3: Run comparison of all methods
        print("\nOption 3: Comparing all evaluation methods")
        comparison_results = compare_evaluation_methods(
            parsed_abstracts_file=str(parsed_abstracts_file),
            output_dir=output_dir + "/comparison",
            model=model,
            max_abstracts=max_abstracts_for_testing
        )
        
        print("\nAll evaluations completed successfully!")
        print("You can now review the results in the generated Excel and JSON files.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
