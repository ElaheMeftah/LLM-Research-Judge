"""Full-schema abstract evaluation workflow."""

import json
from pathlib import Path

from .enhanced_schemas import (
    abstract_criteria_definer, 
    prepare_abstracts_for_evaluation,
    create_evaluation_output
)
from .generator import openai_response_generator
from .result_converter import convert_dict_of_lists_to_list_of_dicts


def create_evaluation_prompt():
    """Return the prompt used for full-schema evaluation."""
    prompt = """
    You are an expert reviewer for a medical conference specializing in infectious diseases and microbiology. 
    Your task is to objectively, fairly, and critically evaluate the following research abstract according to the provided criteria.
    
    Please evaluate the abstract thoroughly and provide scores for each criterion on a scale of 1-10 
    (where 1 is very poor and 10 is excellent) unless otherwise specified.
    
    For boolean fields (like study_type_appropriate, plagiarism_free, etc.), please use true or false.
    For text fields (like comments_for_improvement), provide specific constructive feedback.
    For accept_or_reject, use one of: "Rejected", "Accepted_Poster", or "Accepted_Oral".
    
    Abstract to evaluate:
    {intro_text}
    
    Please provide your evaluation in the exact JSON format specified below:
    {format_instructions}
    
    """
    
    return prompt


def run_complete_evaluation(parsed_abstracts_file: str,
                          output_dir: str = "evaluation_results",
                          model: str = "gpt-4.1-mini",
                          user_criteria: dict = None,
                          criteria_to_keep: list = None,
                          max_abstracts: int = None,
                          save_dict_of_lists: bool = False,
                          return_format: str = "list_of_dicts") -> tuple[str, str]:
    """Run the full-schema evaluation workflow."""
    
    print("Starting complete abstract evaluation workflow...")
    print(f"Using model: {model}")
    
    # Load parsed abstracts
    with open(parsed_abstracts_file, 'r', encoding='utf-8') as f:
        parsed_abstracts = json.load(f)
    
    if max_abstracts:
        parsed_abstracts = parsed_abstracts[:max_abstracts]
        print(f"Limited to first {max_abstracts} abstracts for testing")
    
    print(f"Loaded {len(parsed_abstracts)} abstracts for evaluation")
    
    # Get the parser with enhanced schema
    parser = abstract_criteria_definer(user_criteria=user_criteria, 
                                     criteria_to_keep=criteria_to_keep)
    
    print(f"Created evaluation schema with {len(parser.pydantic_object.__fields__)} criteria")
    
    # Prepare data for evaluation
    evaluation_data = prepare_abstracts_for_evaluation(parsed_abstracts)
    
    # Create evaluation prompt
    prompt = create_evaluation_prompt()
    
    print("Starting LLM evaluation...")
    print("This may take several minutes depending on the number of abstracts...")
    
    # Run LLM evaluation
    results = openai_response_generator(
        data=evaluation_data,
        prompt=prompt,
        parser=parser,
        verbose=True,
        model=model,
        max_tokens=4000
    )
    
    print("LLM evaluation completed!")
    
    try:
        # Convert results to list of dictionaries format
        results_list_of_dicts = convert_dict_of_lists_to_list_of_dicts(results)
        
        # Prepare output directory
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Always create Excel file from dict_of_lists format (for compatibility with create_evaluation_output)
        excel_path, _ = create_evaluation_output(
            results, 
            output_dir=output_dir,
            base_filename=f"abstract_evaluation_{model.replace('-', '_')}"
        )
        
        # Save list of dictionaries format (default primary format)
        list_format_json_path = output_dir_path / f"abstract_evaluation_{model.replace('-', '_')}_list_format.json"
        with open(list_format_json_path, 'w', encoding='utf-8') as f:
            json.dump(results_list_of_dicts, f, ensure_ascii=False, indent=2)
        
        # Conditionally save dict of lists format
        dict_format_json_path = None
        if save_dict_of_lists or return_format == "dict_of_lists":
            dict_format_json_path = output_dir_path / f"abstract_evaluation_{model.replace('-', '_')}_dict_format.json"
            with open(dict_format_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Print results
        print(f"Results saved to:")
        print(f"  Excel: {excel_path}")
        print(f"  JSON (list format): {list_format_json_path}")
        if dict_format_json_path:
            print(f"  JSON (dict format): {dict_format_json_path}")
        
        # Return paths based on requested format
        if return_format == "dict_of_lists":
            primary_json = str(dict_format_json_path) if dict_format_json_path else str(list_format_json_path)
        else:
            primary_json = str(list_format_json_path)
        
        return excel_path, primary_json
        
    except Exception as e:
        print(f"Error during result processing: {e}")
        # Fallback to original behavior
        excel_path, json_path = create_evaluation_output(
            results, 
            output_dir=output_dir,
            base_filename=f"abstract_evaluation_{model.replace('-', '_')}"
        )
        print(f"Results saved to:")
        print(f"  Excel: {excel_path}")
        print(f"  JSON: {json_path}")
        return excel_path, json_path


def main():
    """Run a small local example."""
    
    # Configuration
    parsed_abstracts_file = Path(__file__).parent.parent / "parsed_abstracts.json"
    output_dir = "evaluation_results"
    
    # For testing, limit to first 3 abstracts to save API costs
    # Set to None to evaluate all abstracts
    max_abstracts_for_testing = 3
    
    # Model selection
    model = "gpt-4.1-mini"  # Cost-effective option
    
    # Optional: Define custom criteria
    custom_criteria = {
        'clinical_relevance': 'Rate the clinical relevance of this research',
        'methodology_innovation': 'Assess the innovation in research methodology'
    }
    
    # Optional: Use only specific criteria
    # criteria_subset = ['title', 'overall_evaluation', 'accept_or_reject', 
    #                   'comments_for_improvement', 'innovation', 'scientifically_sound']
    criteria_subset = None  # Use all criteria
    
    try:
        if not parsed_abstracts_file.exists():
            print(f"Error: {parsed_abstracts_file} not found")
            print("Please run the abstract parser first to generate parsed abstracts")
            return
        
        # Run complete evaluation
        excel_path, json_path = run_complete_evaluation(
            parsed_abstracts_file=str(parsed_abstracts_file),
            output_dir=output_dir,
            model=model,
            user_criteria=custom_criteria,
            criteria_to_keep=criteria_subset,
            max_abstracts=max_abstracts_for_testing
        )
        
        print("\nEvaluation completed successfully!")
        print("You can now review the results in the generated Excel and JSON files.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
