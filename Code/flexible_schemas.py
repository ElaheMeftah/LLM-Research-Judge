"""Alternative full, split, numeric, and categorical evaluation schemas."""

from typing import Optional, Literal
from pydantic import BaseModel, Field, create_model
from langchain.output_parsers import PydanticOutputParser
import pandas as pd
from pathlib import Path


# Define the categorical rating type
CategoryRating = Literal['low', 'medium', 'high']


def abstract_criteria_full_numeric(user_criteria: dict = None,
                                 criteria_to_keep: list = None) -> PydanticOutputParser:
    """Build the full numeric evaluation parser."""
    
    # A) PRE-EXTRACTED FIELDS (from abstract_screen and abstract_parser)
    pre_extracted_fields = {
        'abstract_code': (Optional[str], Field(None, 
            description="Unique identifier for the abstract (P1, O1, A1, etc.)")),
        'fulltext': (Optional[str], Field(None, 
            description="Complete structured text of the abstract")),
        'title': (Optional[str], Field(None, 
            description="Title of the research")),
        'authors': (Optional[str], Field(None, 
            description="Author names and affiliation references")),
        'affiliations': (Optional[str], Field(None, 
            description="Institutional affiliations of authors")),
        'introduction': (Optional[str], Field(None, 
            description="Introduction/background section of the abstract")),
        'methods': (Optional[str], Field(None, 
            description="Methods/methodology section of the abstract")),
        'results': (Optional[str], Field(None, 
            description="Results section of the abstract")),
        'conclusions': (Optional[str], Field(None, 
            description="Conclusions section of the abstract")),
        'keywords': (Optional[str], Field(None, 
            description="Keywords or key terms")),
        
        # Screening criteria (boolean fields)
        'study_type_appropriate': (Optional[bool], Field(None, 
            description="NOT a case report or review study")),
        'plagiarism_free': (Optional[bool], Field(None, 
            description="No plagiarism or duplication detected")),
        'human_written': (Optional[bool], Field(None, 
            description="Non-AI-written content")),
        'english_language': (Optional[bool], Field(None, 
            description="Written in English language")),
    }
    
    # B) LLM-GENERATED EVALUATION CRITERIA (1-10 scale)
    llm_evaluation_fields = {
        # Content Quality (1-10 scale)
        'title_quality': (Optional[int], Field(None, ge=1, le=10,
            description="Quality and appropriateness of the title")),
        'relevance': (Optional[int], Field(None, ge=1, le=10,
            description="Relevance to the field and conference theme")),
        'about_congress_special_theme': (Optional[bool], Field(None,
            description="Addresses the special theme of the congress")),
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
            description="Based on multiple centers/institutions according to affiliations or methods")),
        
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
        
        # Structure and Presentation (1-10 scale)
        'structured_format': (Optional[int], Field(None, ge=1, le=10,
            description="Logical abstract structure (Background-Methods-Results-Conclusion)")),
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
            description="Specific suggestions and recommendations for improvement")),
        'accept_or_reject': (Optional[str], Field(None,
            description="Final recommendation: 'rejected', 'accepted_poster', or 'accepted_oral'"))
    }
    
    # Combine all criteria
    all_system_criteria = {**pre_extracted_fields, **llm_evaluation_fields}
    
    # Filter criteria if specified
    if criteria_to_keep:
        selected_criteria = {
            key: val for key, val in all_system_criteria.items() 
            if key in criteria_to_keep
        }
    else:
        selected_criteria = all_system_criteria
    
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


def abstract_criteria_full_categorical(user_criteria: dict = None,
                                     criteria_to_keep: list = None) -> PydanticOutputParser:
    """Build the full categorical evaluation parser."""
    
    # A) PRE-EXTRACTED FIELDS (same as numeric version)
    pre_extracted_fields = {
        'abstract_code': (Optional[str], Field(None, 
            description="Unique identifier for the abstract (P1, O1, A1, etc.)")),
        'fulltext': (Optional[str], Field(None, 
            description="Complete structured text of the abstract")),
        'title': (Optional[str], Field(None, 
            description="Title of the research")),
        'authors': (Optional[str], Field(None, 
            description="Author names and affiliation references")),
        'affiliations': (Optional[str], Field(None, 
            description="Institutional affiliations of authors")),
        'introduction': (Optional[str], Field(None, 
            description="Introduction/background section of the abstract")),
        'methods': (Optional[str], Field(None, 
            description="Methods/methodology section of the abstract")),
        'results': (Optional[str], Field(None, 
            description="Results section of the abstract")),
        'conclusions': (Optional[str], Field(None, 
            description="Conclusions section of the abstract")),
        'keywords': (Optional[str], Field(None, 
            description="Keywords or key terms")),
        
        # Screening criteria (boolean fields)
        'study_type_appropriate': (Optional[bool], Field(None, 
            description="NOT a case report or review study")),
        'plagiarism_free': (Optional[bool], Field(None, 
            description="No plagiarism or duplication detected")),
        'human_written': (Optional[bool], Field(None, 
            description="Non-AI-written content")),
        'english_language': (Optional[bool], Field(None, 
            description="Written in English language")),
    }
    
    # B) LLM-GENERATED EVALUATION CRITERIA (categorical scale)
    llm_evaluation_fields = {
        # Content Quality (low/medium/high scale)
        'title_quality': (Optional[CategoryRating], Field(None,
            description="Quality and appropriateness of the title (low/medium/high)")),
        'relevance': (Optional[CategoryRating], Field(None,
            description="Relevance to the field and conference theme (low/medium/high)")),
        'about_congress_special_theme': (Optional[bool], Field(None,
            description="Addresses the special theme of the congress")),
        'innovation': (Optional[CategoryRating], Field(None,
            description="Novelty and innovative aspects of the research (low/medium/high)")),
        'global_impact': (Optional[CategoryRating], Field(None,
            description="Addressing critical gaps in global/public health as stated by WHO (low/medium/high)")),
        'equity': (Optional[CategoryRating], Field(None,
            description="Addressing social determinants of infectious disease burden (low/medium/high)")),
        'scientifically_sound': (Optional[CategoryRating], Field(None,
            description="Scientific rigor and validity of the research (low/medium/high)")),
        'ethically_sound': (Optional[CategoryRating], Field(None,
            description="Ethical considerations and compliance (low/medium/high)")),
        'multidisciplinary': (Optional[CategoryRating], Field(None,
            description="Interdisciplinary approach and collaboration (low/medium/high)")),
        'multicenter': (Optional[bool], Field(None,
            description="Based on multiple centers/institutions according to affiliations or methods")),
        
        # Writing Quality (low/medium/high scale)
        'context_and_background': (Optional[CategoryRating], Field(None,
            description="Summarizes what is already known and defines key terms for topic understanding (low/medium/high)")),
        'knowledge_gap_identification': (Optional[CategoryRating], Field(None,
            description="Clearly states what remains unknown and why filling the gap is important (low/medium/high)")),
        'rationale_and_significance': (Optional[CategoryRating], Field(None,
            description="Explains study novelty and potential impact on science, clinical care, or policy (low/medium/high)")),
        'logical_flow_and_structure': (Optional[CategoryRating], Field(None,
            description="Each sentence and paragraph leads logically to the next (low/medium/high)")),
        'tone_and_readability': (Optional[CategoryRating], Field(None,
            description="Precise, engaging language that avoids unnecessary jargon (low/medium/high)")),
        'language_and_grammar': (Optional[CategoryRating], Field(None,
            description="Correctness, clarity, fluency and adherence to scientific writing standards (low/medium/high)")),
        'human_generated': (Optional[CategoryRating], Field(None,
            description="How human vs AI-generated the abstract appears (low=AI-like, high=human-like)")),
        'length': (Optional[CategoryRating], Field(None,
            description="Appropriateness of abstract length (standard: 150-300 words) (low/medium/high)")),
        'keyword_quality': (Optional[CategoryRating], Field(None,
            description="Quality and relevance of provided keywords (low/medium/high)")),
        
        # Methodological Quality (low/medium/high scale)
        'appropriate_study_design': (Optional[CategoryRating], Field(None,
            description="Study design is suitable for the research question (low/medium/high)")),
        'methodological_transparency': (Optional[CategoryRating], Field(None,
            description="Methods are clearly described and transparent (low/medium/high)")),
        'proper_sampling': (Optional[CategoryRating], Field(None,
            description="Sample size is properly calculated and described (low/medium/high)")),
        'controls_comparators': (Optional[CategoryRating], Field(None,
            description="Appropriate controls or comparators are used when applicable (low/medium/high)")),
        'statistical_analysis': (Optional[CategoryRating], Field(None,
            description="Statistical methods are valid and properly applied (low/medium/high)")),
        'bias_and_confounding_control': (Optional[CategoryRating], Field(None,
            description="Adequate control of bias and confounding factors (low/medium/high)")),
        'outcome_reporting': (Optional[CategoryRating], Field(None,
            description="Outcomes quantified with proper reporting, not just p-values (low/medium/high)")),
        'result_plausibility': (Optional[CategoryRating], Field(None,
            description="Results are plausible and reliable (low/medium/high)")),
        'missing_data_management': (Optional[CategoryRating], Field(None,
            description="Appropriate handling of missing data (low/medium/high)")),
        'results_and_text_concordance': (Optional[CategoryRating], Field(None,
            description="Consistency between reported results and narrative text (low/medium/high)")),
        
        # Structure and Presentation (low/medium/high scale)
        'structured_format': (Optional[CategoryRating], Field(None,
            description="Logical abstract structure (Background-Methods-Results-Conclusion) (low/medium/high)")),
        'objectives': (Optional[CategoryRating], Field(None,
            description="Research aims and objectives are clearly stated (low/medium/high)")),
        'conclusion': (Optional[CategoryRating], Field(None,
            description="Conclusions are supported by the presented results (low/medium/high)")),
        'translational_potential_explanation': (Optional[CategoryRating], Field(None,
            description="Clinical or practical implications are discussed (low/medium/high)")),
        'future_directions': (Optional[CategoryRating], Field(None,
            description="Future research directions are appropriately discussed (low/medium/high)")),
        
        # Overall Assessment
        'overall_evaluation': (Optional[CategoryRating], Field(None,
            description="Overall quality assessment of the abstract (low/medium/high)")),
        'comments_for_improvement': (Optional[str], Field(None,
            description="Specific suggestions and recommendations for improvement")),
        'accept_or_reject': (Optional[str], Field(None,
            description="Final recommendation: 'rejected', 'accepted_poster', or 'accepted_oral'"))
    }
    
    # Combine all criteria
    all_system_criteria = {**pre_extracted_fields, **llm_evaluation_fields}
    
    # Filter criteria if specified
    if criteria_to_keep:
        selected_criteria = {
            key: val for key, val in all_system_criteria.items() 
            if key in criteria_to_keep
        }
    else:
        selected_criteria = all_system_criteria
    
    # Add user-defined criteria if provided
    if user_criteria:
        user_defined_criteria = {}
        for key, value in user_criteria.items():
            if isinstance(value, str):
                # Text criteria
                user_defined_criteria[key] = (Optional[str], Field(None, description=value))
            else:
                # Assume categorical criteria
                user_defined_criteria[key] = (Optional[CategoryRating], Field(None, description=str(value)))
    else:
        user_defined_criteria = {}
    
    # Combine all criteria
    final_criteria = {**selected_criteria, **user_defined_criteria}
    
    # Create the Pydantic model
    pydantic_model = create_model('AbstractEvaluationCriteriaCategorical', **final_criteria)
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    
    return parser


def abstract_criteria_split_part_a_numeric(user_criteria: dict = None,
                                         criteria_to_keep: list = None) -> PydanticOutputParser:
    """Build the numeric parser for content and writing."""
    
    # A) PRE-EXTRACTED FIELDS
    pre_extracted_fields = {
        'abstract_code': (Optional[str], Field(None, 
            description="Unique identifier for the abstract (P1, O1, A1, etc.)")),
        'fulltext': (Optional[str], Field(None, 
            description="Complete structured text of the abstract")),
        'title': (Optional[str], Field(None, 
            description="Title of the research")),
        'authors': (Optional[str], Field(None, 
            description="Author names and affiliation references")),
        'affiliations': (Optional[str], Field(None, 
            description="Institutional affiliations of authors")),
        'introduction': (Optional[str], Field(None, 
            description="Introduction/background section of the abstract")),
        'methods': (Optional[str], Field(None, 
            description="Methods/methodology section of the abstract")),
        'results': (Optional[str], Field(None, 
            description="Results section of the abstract")),
        'conclusions': (Optional[str], Field(None, 
            description="Conclusions section of the abstract")),
        'keywords': (Optional[str], Field(None, 
            description="Keywords or key terms")),
        
        # Screening criteria (boolean fields)
        'study_type_appropriate': (Optional[bool], Field(None, 
            description="NOT a case report or review study")),
        'plagiarism_free': (Optional[bool], Field(None, 
            description="No plagiarism or duplication detected")),
        'human_written': (Optional[bool], Field(None, 
            description="Non-AI-written content")),
        'english_language': (Optional[bool], Field(None, 
            description="Written in English language")),
    }
    
    # B) CONTENT QUALITY + WRITING QUALITY
    content_and_writing_fields = {
        # Content Quality (1-10 scale)
        'title_quality': (Optional[int], Field(None, ge=1, le=10,
            description="Quality and appropriateness of the title")),
        'relevance': (Optional[int], Field(None, ge=1, le=10,
            description="Relevance to the field and conference theme")),
        'about_congress_special_theme': (Optional[bool], Field(None,
            description="Addresses the special theme of the congress")),
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
            description="Based on multiple centers/institutions according to affiliations or methods")),
        
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
    }
    
    # Combine Part A criteria
    part_a_criteria = {**pre_extracted_fields, **content_and_writing_fields}
    
    # Filter criteria if specified
    if criteria_to_keep:
        selected_criteria = {
            key: val for key, val in part_a_criteria.items() 
            if key in criteria_to_keep
        }
    else:
        selected_criteria = part_a_criteria
    
    # Add user-defined criteria if provided
    if user_criteria:
        user_defined_criteria = {}
        for key, value in user_criteria.items():
            if isinstance(value, str):
                user_defined_criteria[key] = (Optional[str], Field(None, description=value))
            else:
                user_defined_criteria[key] = (Optional[int], Field(None, ge=1, le=10, description=str(value)))
    else:
        user_defined_criteria = {}
    
    # Combine all criteria
    final_criteria = {**selected_criteria, **user_defined_criteria}
    
    # Create the Pydantic model
    pydantic_model = create_model('AbstractEvaluationPartA', **final_criteria)
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    
    return parser


def abstract_criteria_split_part_b_numeric(user_criteria: dict = None,
                                         criteria_to_keep: list = None) -> PydanticOutputParser:
    """Build the numeric parser for methods and final assessment."""
    
    # Essential identifier for merging
    identifier_field = {
        'abstract_code': (Optional[str], Field(None, 
            description="Unique identifier for the abstract (P1, O1, A1, etc.) - for merging results")),
    }
    
    # METHODOLOGICAL QUALITY + STRUCTURE + OVERALL ASSESSMENT
    method_structure_overall_fields = {
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
        
        # Structure and Presentation (1-10 scale)
        'structured_format': (Optional[int], Field(None, ge=1, le=10,
            description="Logical abstract structure (Background-Methods-Results-Conclusion)")),
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
            description="Specific suggestions and recommendations for improvement")),
        'accept_or_reject': (Optional[str], Field(None,
            description="Final recommendation: 'rejected', 'accepted_poster', or 'accepted_oral'"))
    }
    
    # Combine Part B criteria
    part_b_criteria = {**identifier_field, **method_structure_overall_fields}
    
    # Filter criteria if specified
    if criteria_to_keep:
        selected_criteria = {
            key: val for key, val in part_b_criteria.items() 
            if key in criteria_to_keep
        }
    else:
        selected_criteria = part_b_criteria
    
    # Add user-defined criteria if provided
    if user_criteria:
        user_defined_criteria = {}
        for key, value in user_criteria.items():
            if isinstance(value, str):
                user_defined_criteria[key] = (Optional[str], Field(None, description=value))
            else:
                user_defined_criteria[key] = (Optional[int], Field(None, ge=1, le=10, description=str(value)))
    else:
        user_defined_criteria = {}
    
    # Combine all criteria
    final_criteria = {**selected_criteria, **user_defined_criteria}
    
    # Create the Pydantic model
    pydantic_model = create_model('AbstractEvaluationPartB', **final_criteria)
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    
    return parser


def abstract_criteria_split_part_a_categorical(user_criteria: dict = None,
                                             criteria_to_keep: list = None) -> PydanticOutputParser:
    """Build the categorical parser for content and writing."""
    
    # A) PRE-EXTRACTED FIELDS (same as numeric)
    pre_extracted_fields = {
        'abstract_code': (Optional[str], Field(None, 
            description="Unique identifier for the abstract (P1, O1, A1, etc.)")),
        'fulltext': (Optional[str], Field(None, 
            description="Complete structured text of the abstract")),
        'title': (Optional[str], Field(None, 
            description="Title of the research")),
        'authors': (Optional[str], Field(None, 
            description="Author names and affiliation references")),
        'affiliations': (Optional[str], Field(None, 
            description="Institutional affiliations of authors")),
        'introduction': (Optional[str], Field(None, 
            description="Introduction/background section of the abstract")),
        'methods': (Optional[str], Field(None, 
            description="Methods/methodology section of the abstract")),
        'results': (Optional[str], Field(None, 
            description="Results section of the abstract")),
        'conclusions': (Optional[str], Field(None, 
            description="Conclusions section of the abstract")),
        'keywords': (Optional[str], Field(None, 
            description="Keywords or key terms")),
        
        # Screening criteria (boolean fields)
        'study_type_appropriate': (Optional[bool], Field(None, 
            description="NOT a case report or review study")),
        'plagiarism_free': (Optional[bool], Field(None, 
            description="No plagiarism or duplication detected")),
        'human_written': (Optional[bool], Field(None, 
            description="Non-AI-written content")),
        'english_language': (Optional[bool], Field(None, 
            description="Written in English language")),
    }
    
    # B) CONTENT QUALITY + WRITING QUALITY (categorical)
    content_and_writing_fields = {
        # Content Quality (low/medium/high scale)
        'title_quality': (Optional[CategoryRating], Field(None,
            description="Quality and appropriateness of the title (low/medium/high)")),
        'relevance': (Optional[CategoryRating], Field(None,
            description="Relevance to the field and conference theme (low/medium/high)")),
        'about_congress_special_theme': (Optional[bool], Field(None,
            description="Addresses the special theme of the congress")),
        'innovation': (Optional[CategoryRating], Field(None,
            description="Novelty and innovative aspects of the research (low/medium/high)")),
        'global_impact': (Optional[CategoryRating], Field(None,
            description="Addressing critical gaps in global/public health as stated by WHO (low/medium/high)")),
        'equity': (Optional[CategoryRating], Field(None,
            description="Addressing social determinants of infectious disease burden (low/medium/high)")),
        'scientifically_sound': (Optional[CategoryRating], Field(None,
            description="Scientific rigor and validity of the research (low/medium/high)")),
        'ethically_sound': (Optional[CategoryRating], Field(None,
            description="Ethical considerations and compliance (low/medium/high)")),
        'multidisciplinary': (Optional[CategoryRating], Field(None,
            description="Interdisciplinary approach and collaboration (low/medium/high)")),
        'multicenter': (Optional[bool], Field(None,
            description="Based on multiple centers/institutions according to affiliations or methods")),
        
        # Writing Quality (low/medium/high scale)
        'context_and_background': (Optional[CategoryRating], Field(None,
            description="Summarizes what is already known and defines key terms for topic understanding (low/medium/high)")),
        'knowledge_gap_identification': (Optional[CategoryRating], Field(None,
            description="Clearly states what remains unknown and why filling the gap is important (low/medium/high)")),
        'rationale_and_significance': (Optional[CategoryRating], Field(None,
            description="Explains study novelty and potential impact on science, clinical care, or policy (low/medium/high)")),
        'logical_flow_and_structure': (Optional[CategoryRating], Field(None,
            description="Each sentence and paragraph leads logically to the next (low/medium/high)")),
        'tone_and_readability': (Optional[CategoryRating], Field(None,
            description="Precise, engaging language that avoids unnecessary jargon (low/medium/high)")),
        'language_and_grammar': (Optional[CategoryRating], Field(None,
            description="Correctness, clarity, fluency and adherence to scientific writing standards (low/medium/high)")),
        'human_generated': (Optional[CategoryRating], Field(None,
            description="How human vs AI-generated the abstract appears (low=AI-like, high=human-like)")),
        'length': (Optional[CategoryRating], Field(None,
            description="Appropriateness of abstract length (standard: 150-300 words) (low/medium/high)")),
        'keyword_quality': (Optional[CategoryRating], Field(None,
            description="Quality and relevance of provided keywords (low/medium/high)")),
    }
    
    # Combine Part A criteria
    part_a_criteria = {**pre_extracted_fields, **content_and_writing_fields}
    
    # Filter criteria if specified
    if criteria_to_keep:
        selected_criteria = {
            key: val for key, val in part_a_criteria.items() 
            if key in criteria_to_keep
        }
    else:
        selected_criteria = part_a_criteria
    
    # Add user-defined criteria if provided
    if user_criteria:
        user_defined_criteria = {}
        for key, value in user_criteria.items():
            if isinstance(value, str):
                user_defined_criteria[key] = (Optional[str], Field(None, description=value))
            else:
                user_defined_criteria[key] = (Optional[CategoryRating], Field(None, description=str(value)))
    else:
        user_defined_criteria = {}
    
    # Combine all criteria
    final_criteria = {**selected_criteria, **user_defined_criteria}
    
    # Create the Pydantic model
    pydantic_model = create_model('AbstractEvaluationPartACategorical', **final_criteria)
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    
    return parser


def abstract_criteria_split_part_b_categorical(user_criteria: dict = None,
                                             criteria_to_keep: list = None) -> PydanticOutputParser:
    """Build the categorical parser for methods and final assessment."""
    
    # Essential identifier for merging
    identifier_field = {
        'abstract_code': (Optional[str], Field(None, 
            description="Unique identifier for the abstract (P1, O1, A1, etc.) - for merging results")),
    }
    
    # METHODOLOGICAL QUALITY + STRUCTURE + OVERALL ASSESSMENT (categorical)
    method_structure_overall_fields = {
        # Methodological Quality (low/medium/high scale)
        'appropriate_study_design': (Optional[CategoryRating], Field(None,
            description="Study design is suitable for the research question (low/medium/high)")),
        'methodological_transparency': (Optional[CategoryRating], Field(None,
            description="Methods are clearly described and transparent (low/medium/high)")),
        'proper_sampling': (Optional[CategoryRating], Field(None,
            description="Sample size is properly calculated and described (low/medium/high)")),
        'controls_comparators': (Optional[CategoryRating], Field(None,
            description="Appropriate controls or comparators are used when applicable (low/medium/high)")),
        'statistical_analysis': (Optional[CategoryRating], Field(None,
            description="Statistical methods are valid and properly applied (low/medium/high)")),
        'bias_and_confounding_control': (Optional[CategoryRating], Field(None,
            description="Adequate control of bias and confounding factors (low/medium/high)")),
        'outcome_reporting': (Optional[CategoryRating], Field(None,
            description="Outcomes quantified with proper reporting, not just p-values (low/medium/high)")),
        'result_plausibility': (Optional[CategoryRating], Field(None,
            description="Results are plausible and reliable (low/medium/high)")),
        'missing_data_management': (Optional[CategoryRating], Field(None,
            description="Appropriate handling of missing data (low/medium/high)")),
        'results_and_text_concordance': (Optional[CategoryRating], Field(None,
            description="Consistency between reported results and narrative text (low/medium/high)")),
        
        # Structure and Presentation (low/medium/high scale)
        'structured_format': (Optional[CategoryRating], Field(None,
            description="Logical abstract structure (Background-Methods-Results-Conclusion) (low/medium/high)")),
        'objectives': (Optional[CategoryRating], Field(None,
            description="Research aims and objectives are clearly stated (low/medium/high)")),
        'conclusion': (Optional[CategoryRating], Field(None,
            description="Conclusions are supported by the presented results (low/medium/high)")),
        'translational_potential_explanation': (Optional[CategoryRating], Field(None,
            description="Clinical or practical implications are discussed (low/medium/high)")),
        'future_directions': (Optional[CategoryRating], Field(None,
            description="Future research directions are appropriately discussed (low/medium/high)")),
        
        # Overall Assessment
        'overall_evaluation': (Optional[CategoryRating], Field(None,
            description="Overall quality assessment of the abstract (low/medium/high)")),
        'comments_for_improvement': (Optional[str], Field(None,
            description="Specific suggestions and recommendations for improvement")),
        'accept_or_reject': (Optional[str], Field(None,
            description="Final recommendation: 'rejected', 'accepted_poster', or 'accepted_oral'"))
    }
    
    # Combine Part B criteria
    part_b_criteria = {**identifier_field, **method_structure_overall_fields}
    
    # Filter criteria if specified
    if criteria_to_keep:
        selected_criteria = {
            key: val for key, val in part_b_criteria.items() 
            if key in criteria_to_keep
        }
    else:
        selected_criteria = part_b_criteria
    
    # Add user-defined criteria if provided
    if user_criteria:
        user_defined_criteria = {}
        for key, value in user_criteria.items():
            if isinstance(value, str):
                user_defined_criteria[key] = (Optional[str], Field(None, description=value))
            else:
                user_defined_criteria[key] = (Optional[CategoryRating], Field(None, description=str(value)))
    else:
        user_defined_criteria = {}
    
    # Combine all criteria
    final_criteria = {**selected_criteria, **user_defined_criteria}
    
    # Create the Pydantic model
    pydantic_model = create_model('AbstractEvaluationPartBCategorical', **final_criteria)
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    
    return parser


def merge_split_evaluation_results(part_a_results: dict, part_b_results: dict) -> dict:
    """Merge Part A and Part B outputs into one result set."""
    
    # Start with Part A results as the base
    merged_results = part_a_results.copy()
    
    # Add Part B fields (excluding abstract_code which should already be in Part A)
    for key, values in part_b_results.items():
        if key != 'abstract_code' and key not in merged_results:
            merged_results[key] = values
        elif key == 'abstract_code':
            # Verify that abstract codes match between Part A and Part B
            if 'abstract_code' in merged_results:
                part_a_codes = merged_results['abstract_code']
                part_b_codes = values
                if part_a_codes != part_b_codes:
                    print("Warning: Abstract codes don't match between Part A and Part B")
                    print(f"Part A codes: {part_a_codes}")
                    print(f"Part B codes: {part_b_codes}")
    
    return merged_results


def convert_categorical_to_numeric(results: dict, 
                                 categorical_mapping: dict = None) -> dict:
    """Map low, medium, and high ratings to numbers."""
    
    if categorical_mapping is None:
        categorical_mapping = {
            'low': 3,
            'medium': 6, 
            'high': 9
        }
    
    converted_results = results.copy()
    
    for key, values in converted_results.items():
        if isinstance(values, list):
            converted_values = []
            for value in values:
                if value in categorical_mapping:
                    converted_values.append(categorical_mapping[value])
                else:
                    converted_values.append(value)  # Keep non-categorical values as is
            converted_results[key] = converted_values
    
    return converted_results


def convert_numeric_to_categorical(results: dict,
                                 numeric_mapping: dict = None) -> dict:
    """Map 1-10 scores to low, medium, or high ratings."""
    
    if numeric_mapping is None:
        def score_to_category(score):
            if score is None:
                return None
            if score <= 3:
                return 'low'
            elif score <= 7:
                return 'medium'
            else:
                return 'high'
    else:
        def score_to_category(score):
            return numeric_mapping.get(score, score)
    
    converted_results = results.copy()
    
    for key, values in converted_results.items():
        if isinstance(values, list):
            converted_values = []
            for value in values:
                if isinstance(value, int) and 1 <= value <= 10:
                    converted_values.append(score_to_category(value))
                else:
                    converted_values.append(value)  # Keep non-numeric values as is
            converted_results[key] = converted_values
    
    return converted_results


# For backwards compatibility, provide the original function
def abstract_criteria_definer(user_criteria: dict = None,
                            criteria_to_keep: list = None) -> PydanticOutputParser:
    """Build the default abstract evaluation parser."""
    return abstract_criteria_full_numeric(user_criteria, criteria_to_keep)


if __name__ == "__main__":
    # Example usage
    print("Testing flexible schemas...")
    
    # Test all schema variations
    schemas = {
        'full_numeric': abstract_criteria_full_numeric(),
        'full_categorical': abstract_criteria_full_categorical(),
        'split_a_numeric': abstract_criteria_split_part_a_numeric(),
        'split_b_numeric': abstract_criteria_split_part_b_numeric(),
        'split_a_categorical': abstract_criteria_split_part_a_categorical(),
        'split_b_categorical': abstract_criteria_split_part_b_categorical(),
    }
    
    for name, parser in schemas.items():
        print(f"{name}: {len(parser.pydantic_object.__fields__)} criteria")
    
    print("\nFlexible schemas created successfully!")
