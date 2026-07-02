
"""Schema helper for introduction-level evaluation."""


from typing import Optional
from pydantic import BaseModel, Field, create_model
from langchain.output_parsers import PydanticOutputParser


# ------------------------------------------
#       Pydantic model for Introduction
# ------------------------------------------

def introduction_criteria_definer(user_criteria: dict = None,
                        base_criteria_to_keep: list = None) -> BaseModel:
    """Build a parser for introduction quality criteria."""

    # modifying base_criteria, if requested by the user
    base_criteria = {
    'title': (Optional[str], Field(
        None,
        description="The title of the article."
    )),
    'context_and_background': (Optional[int], Field(
        None,
        ge=1,
        le=10,
        description="Summarize what is already known and define key terms so readers understand the topic's setting."
    )),
    'knowledge_gap_identification': (Optional[int], Field(
        None,
        ge=1,
        le=10,
        description="After background explanation, clearly state what remains unknown or unresolved and why filling that gap is important."
    )),
    'rationale_and_significance': (Optional[int], Field(
        None,
        ge=1,
        le=10,
        description="Explain the study's novelty and its potential impact on science, clinical care, or policy."
    )),
    'objectives_and_hypotheses': (Optional[int], Field(
        None,
        ge=1,
        le=10,
        description="List specific, testable aims or questions that directly address the identified gap."
    )),
    'logical_flow_and_structure': (Optional[int], Field(
        None,
        ge=1,
        le=10,
        description="Ensure each sentence and paragraph leads logically to the next, building concisely toward the study aims."
    )),
    'citation_quality': (Optional[int], Field(
        None,
        ge=1,
        le=10,
        description="Use up-to-date, relevant, high-quality references and acknowledge conflicting evidence."
    )),
    'tone_and_readability': (Optional[int], Field(
        None,
        ge=1,
        le=10,
        description="Write in precise, engaging language that avoids unnecessary jargon or overstatement."
    )),
    'language_and_grammar': (Optional[int], Field(
        None,
        ge=1,
        le=10,
        description="Evaluate correctness, clarity, and fluency of the writing and adherence to scientific writing methods."
    )),
    'human_generated': (Optional[int], Field(
        None,
        ge=1,
        le=10,
        description="Rate how AI-generated the introduction appears (1 = fully AI-generated, 10 = fully human)."
    )),
    'length': (Optional[int], Field(
        None,
        ge=1,
        le=10,
        description="Evaluate if the introduction is not too long or too short (standard: between 400-700 words)."
    )),
    'overall_evaluation': (Optional[int], Field(
        None,
        ge=1,
        le=10,
        description="Provide a concise judgment of the introduction's overall strengths, weaknesses, and readiness."
    ))
    }

    selected_base_criteria = {}
    if base_criteria_to_keep:
        selected_base_criteria = {
            key: val for key, val in base_criteria.items() if key in base_criteria_to_keep
        }
    else:
        selected_base_criteria = base_criteria

    # defining user_criteria, if given by the user
    if user_criteria is None:
        user_criteria = {}
    user_defined_criteria = {
        key: (Optional[int], Field(None, description=val)) for key, val in user_criteria.items()
    }

    all_criteria = {**selected_base_criteria, **user_defined_criteria}

    pydantic_model = create_model('ModifiedCriteria', **all_criteria)
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    format_instructions = parser.get_format_instructions()

    return parser
