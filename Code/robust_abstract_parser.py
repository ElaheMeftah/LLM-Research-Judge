"""Parsers for congress abstract files."""




# ---------------------------------------------------
#      Specific for Prof. Alborzi Congress 2025
# ---------------------------------------------------

import pandas as pd
import openpyxl
import json
import re
from typing import Dict, List


def parse_abstract_alborzi(abstract_text: str) -> Dict[str, str]:
    """Parse a congress abstract into standard sections."""
    if not abstract_text or pd.isna(abstract_text):
        return {
            'Introduction': '',
            'Methods': '',
            'Results': '',
            'Conclusions': '',
            'Keywords': ''}
    
    sections = {
        'Introduction': '',
        'Methods': '',
        'Results': '',
        'Conclusions': '',
        'Keywords': ''}
    
    intro_match = re.search(r'Introduction:\s*(.*?)(?=Methods:|Results:|Conclusion:|Keywords:|$)', 
                           abstract_text, re.IGNORECASE | re.DOTALL)
    methods_match = re.search(r'Methods:\s*(.*?)(?=Results:|Conclusion:|Keywords:|$)', 
                             abstract_text, re.IGNORECASE | re.DOTALL)
    results_match = re.search(r'Results:\s*(.*?)(?=Conclusion:|Keywords:|$)', 
                             abstract_text, re.IGNORECASE | re.DOTALL)
    conclusion_match = re.search(r'Conclusion:\s*(.*?)(?=Keywords:|$)', 
                                abstract_text, re.IGNORECASE | re.DOTALL)
    keywords_match = re.search(r'Keywords:\s*(.*?)$', 
                              abstract_text, re.IGNORECASE | re.DOTALL)
    
    if intro_match:
        sections['Introduction'] = intro_match.group(1).strip()
    if methods_match:
        sections['Methods'] = methods_match.group(1).strip()
    if results_match:
        sections['Results'] = results_match.group(1).strip()
    if conclusion_match:
        sections['Conclusions'] = conclusion_match.group(1).strip()
    if keywords_match:
        sections['Keywords'] = keywords_match.group(1).strip()
    
    return sections


def excel_to_json(filename: str,
                        output_filename: str = 'parsed_abstracts_alborzi.json') -> List[Dict]:

    df = pd.read_excel(filename)
    results = []
    
    for _, row in df.iterrows():
        parsed_sections = parse_abstract_alborzi(row.get('Abstract', ''))
        fulltext = f"Title: {row.get('Title En', '')} | Affiliations: {row.get('Affiliations', '')} | {row.get('Abstract', '')}"

        article = {
            'abstract_code': row.get('Code', ''),
            'Title': row.get('Title En', ''),
            'Authors': row.get('Authors', ''),
            'Affiliations': row.get('Affiliations', ''),
            'Introduction': parsed_sections['Introduction'],
            'Methods': parsed_sections['Methods'],
            'Results': parsed_sections['Results'],
            'Conclusions': parsed_sections['Conclusions'],
            'Keywords': parsed_sections['Keywords'],
            'Abstract': row.get('Abstract', ''),
            'Fulltext': fulltext
        }
        
        results.append(article)

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results



# ---------------------------------------------------
#             Robust for Congress DOCX Inputs
# ---------------------------------------------------

import re
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx2txt
    DOCX2TXT_AVAILABLE = True
except ImportError:
    DOCX2TXT_AVAILABLE = False

try:
    import textract
    TEXTRACT_AVAILABLE = True
except ImportError:
    TEXTRACT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced regex patterns
INTRO_HDR = r'(?:Introduction(?:\s+and\s+Objectives?)?|Background|Objective[s]?|Aim[s]?|Purpose)'
METHODS_HDR = r'(?:(?:Materials?|Material)\s+and\s+Methods?|Method(?:s|ology)?|Procedures?|Study\s+Design)'
RESULTS_HDR = r'(?:Results?|Findings?|Outcomes?)'
CONCLUSION_HDR = r'(?:Conclusion(?:s)?|Discussion|Concluding\s+remarks?|Summary)'
KEYWORDS_HDR = r'(?:Key\s*words?|Keywords?)'


def extract_text_from_docx(file_path: str) -> str:
    """Read text from a DOCX file."""
    try:
        if TEXTRACT_AVAILABLE:
            try:
                return textract.process(file_path).decode('utf-8')
            except Exception:
                pass
        elif DOCX_AVAILABLE:
            doc = Document(file_path)
            text_parts = []
            for paragraph in doc.paragraphs:
                text_parts.append(paragraph.text)
            return '\n'.join(text_parts)
        elif DOCX2TXT_AVAILABLE:
            return docx2txt.process(file_path)
        else:
            raise ImportError("No DOCX reader available")
    except Exception as e:
        logger.error(f"Error reading DOCX file {file_path}: {e}")
        return ""


def extract_text_from_pdf(file_path: str) -> str:
    """Read text from a PDF file."""
    try:
        if not PDF_AVAILABLE:
            raise ImportError("PyMuPDF not available")
        
        doc = fitz.open(file_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return '\n'.join(text_parts)
    except Exception as e:
        logger.error(f"Error reading PDF file {file_path}: {e}")
        return ""


def extract_text_from_file(file_path: str) -> str:
    """Read text from a supported file type."""
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    if extension == '.docx':
        return extract_text_from_docx(str(file_path))
    elif extension == '.pdf':
        return extract_text_from_pdf(str(file_path))
    elif extension == '.txt':
        return textract.process(file_path).decode('utf-8')
    else:
        logger.warning(f"Unsupported file format: {extension}")
        return ""


def _extract_section(block, header_regex, next_headers_regex):
    """Extract one labeled section from an abstract block."""
    pattern = rf'(?is)\b{header_regex}\s*:?\s*(.*?)\s*(?=\n\s*(?:{next_headers_regex})\s*:|\Z)'
    m = re.search(pattern, block, flags=re.IGNORECASE | re.DOTALL)
    return (m.group(1).strip() if m else "")


def extract_affiliations_robust(block: str) -> str:
    """Find likely affiliation lines in an abstract block."""
    # Strategy 1: Look for numbered lines with institutional keywords
    numbered_affil_pattern = r'^\s*(\d+)[\s:\-\.\)\]]*(.+(?:University|Institute|Hospital|Department|Faculty|School|Center|Centre|Research|Laboratory|Lab|College|Clinic).+)'
    numbered_matches = re.findall(numbered_affil_pattern, block, re.MULTILINE | re.IGNORECASE)
    
    if numbered_matches:
        affiliations = []
        for num, affil in numbered_matches:
            clean_affil = re.sub(r'\s+', ' ', affil.strip())
            if len(clean_affil) > 10:
                affiliations.append(f"{num}. {clean_affil}")
        if affiliations:
            return '; '.join(affiliations)
    
    # Strategy 2: Look for lines starting with institutional keywords
    institutional_pattern = r'^\s*(.+(?:University|Institute|Hospital|Department|Faculty|School|Center|Centre|Research|Laboratory|Lab|College|Clinic).+)'
    institutional_matches = re.findall(institutional_pattern, block, re.MULTILINE | re.IGNORECASE)
    
    if institutional_matches:
        affiliations = []
        for match in institutional_matches:
            clean_affil = re.sub(r'\s+', ' ', match.strip())
            if len(clean_affil) > 15:  # Filter out very short matches
                affiliations.append(clean_affil)
        if affiliations:
            return '; '.join(affiliations[:5])  # Limit to first 5 to avoid noise
    
    return ""


def parse_abstract_corrected(block, code=""):
    """Parse a structured abstract block."""
    # Normalize spacing
    block = re.sub(r'[ \t]+', ' ', block)
    block = re.sub(r'\n{3,}', '\n\n', block)

    # Extract or use provided code
    if not code:
        code_match = re.search(r'(?im)^\s*([PO]\d+)\b', block)
        code = code_match.group(1) if code_match else ""
        # Remove code from block for further processing
        if code_match:
            block = block[code_match.end():].lstrip('\n ')

    # Split into lines for line-by-line analysis
    lines = [line.strip() for line in block.split('\n') if line.strip()]
    
    if not lines:
        return {
            "abstract_code": code or None,
            "Title": None, "Authors": None, "Affiliations": None,
            "Introduction": None, "Methods": None, "Results": None,
            "Conclusions": None, "Keywords": None, "Fulltext": None,
        }

    title = ""
    authors = ""
    affiliations = ""
    
    # Strategy: Sequential line analysis
    line_idx = 0
    
    # 1. Extract title (first line that doesn't look like authors/affiliations/sections)
    while line_idx < len(lines):
        line = lines[line_idx]
        
        # Skip if it looks like a section header
        if re.search(rf'(?i)^\s*(?:{INTRO_HDR}|{METHODS_HDR}|{RESULTS_HDR}|{CONCLUSION_HDR}|{KEYWORDS_HDR})\s*:', line):
            break
            
        # Skip if it looks like affiliations (starts with number + institution)
        if re.search(r'^\s*\d+[\s:\-\.\)\]]*(?:University|Institute|Hospital|Department|Faculty|School|Center|Centre|Research|Laboratory|Lab|College|Clinic)', line, re.IGNORECASE):
            break
            
        # Skip if it looks like authors (names with numbers/symbols)
        if re.search(r'^[A-Z][a-z]+ [A-Z][a-z]+.*[0-9*†‡¹²³⁴⁵]+', line):
            break
            
        # This should be the title
        title = line
        line_idx += 1
        break
    
    # 2. Extract authors (next line that looks like author names)
    while line_idx < len(lines):
        line = lines[line_idx]
        
        # Skip if it looks like a section header
        if re.search(rf'(?i)^\s*(?:{INTRO_HDR}|{METHODS_HDR}|{RESULTS_HDR}|{CONCLUSION_HDR}|{KEYWORDS_HDR})\s*:', line):
            break
            
        # Skip if it looks like affiliations
        if re.search(r'^\s*\d+[\s:\-\.\)\]]*(?:University|Institute|Hospital|Department|Faculty|School|Center|Centre|Research|Laboratory|Lab|College|Clinic)', line, re.IGNORECASE):
            break
            
        # Check if it looks like authors (names with numbers/symbols or multiple names)
        if (re.search(r'^[A-Z][a-z]+ [A-Z][a-z]+.*[0-9*†‡¹²³⁴⁵]+', line) or
            re.search(r'^[A-Z][a-z]+ [A-Z][a-z]+.*,.*[A-Z][a-z]+ [A-Z][a-z]+', line) or
            (len(line.split()) >= 2 and re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', line))):
            authors = line
            line_idx += 1
            break
        
        line_idx += 1
    
    # 3. Extract affiliations (look for institutional lines after authors)
    affiliation_lines = []
    while line_idx < len(lines):
        line = lines[line_idx]
        
        # Stop if we hit section headers
        if re.search(rf'(?i)^\s*(?:{INTRO_HDR}|{METHODS_HDR}|{RESULTS_HDR}|{CONCLUSION_HDR}|{KEYWORDS_HDR})\s*:', line):
            break
            
        # Check if it looks like an affiliation
        if (re.search(r'^\s*\d+[\s:\-\.\)\]]*(?:University|Institute|Hospital|Department|Faculty|School|Center|Centre|Research|Laboratory|Lab|College|Clinic)', line, re.IGNORECASE) or
            re.search(r'(?:University|Institute|Hospital|Department|Faculty|School|Center|Centre|Research|Laboratory|Lab|College|Clinic)', line, re.IGNORECASE)):
            affiliation_lines.append(line)
        
        line_idx += 1
        
        # Stop after finding a few affiliation lines or if we've gone too far
        if len(affiliation_lines) >= 5 or line_idx > 10:
            break
    
    # Process affiliations
    if affiliation_lines:
        # Try to format numbered affiliations properly
        processed_affiliations = []
        for line in affiliation_lines:
            # Check if it starts with a number
            num_match = re.match(r'^\s*(\d+)[\s:\-\.\)\]]*(.+)', line)
            if num_match:
                num, text = num_match.groups()
                processed_affiliations.append(f"{num}. {text.strip()}")
            else:
                processed_affiliations.append(line.strip())
        affiliations = '; '.join(processed_affiliations)
    
    # If we still don't have affiliations, try the robust extraction on the whole block
    if not affiliations:
        affiliations = extract_affiliations_robust(block)

    # 4. Extract sections from the remaining text
    remaining_text = '\n'.join(lines[line_idx:]) if line_idx < len(lines) else ""
    
    # If we didn't find title/authors from line analysis, try the whole block
    if not title and not authors:
        # Fallback: use paragraph-based approach
        paragraphs = re.split(r'\n\s*\n', block)
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
                
            # Skip section headers
            if re.search(rf'(?i)^\s*(?:{INTRO_HDR}|{METHODS_HDR}|{RESULTS_HDR}|{CONCLUSION_HDR}|{KEYWORDS_HDR})\s*:', para):
                remaining_text = '\n\n'.join(paragraphs[i:])
                break
                
            # Skip affiliation paragraphs
            if re.search(r'(?:University|Institute|Hospital|Department|Faculty|School|Center|Centre|Research|Laboratory|Lab|College|Clinic)', para, re.IGNORECASE):
                if not affiliations:
                    affiliations = extract_affiliations_robust(para)
                continue
                
            # First substantial paragraph should be title
            if not title and len(para) > 20:
                title = re.sub(r'\s+', ' ', para)
                continue
                
            # Next paragraph should be authors
            if not authors and len(para) > 10:
                authors = re.sub(r'\s+', ' ', para)
                remaining_text = '\n\n'.join(paragraphs[i+1:])
                break

    # Extract sections from remaining text
    next_all = rf'{INTRO_HDR}|{METHODS_HDR}|{RESULTS_HDR}|{CONCLUSION_HDR}|{KEYWORDS_HDR}'

    intro = _extract_section(remaining_text, INTRO_HDR, next_all)
    methods = _extract_section(remaining_text, METHODS_HDR, next_all)
    results = _extract_section(remaining_text, RESULTS_HDR, next_all)
    conclusions = _extract_section(remaining_text, CONCLUSION_HDR, next_all)

    # Keywords
    kw = _extract_section(remaining_text, KEYWORDS_HDR, next_all)
    keywords = ", ".join([k.strip(" ,.;") for k in re.split(r'[;,]\s*|\s{2,}', kw) if k.strip()]) if kw else ""

    # Create fulltext
    fulltext_parts = []
    if intro:
        fulltext_parts.append(f"Introduction: {intro}")
    if methods:
        fulltext_parts.append(f"Methods: {methods}")
    if results:
        fulltext_parts.append(f"Results: {results}")
    if conclusions:
        fulltext_parts.append(f"Conclusions: {conclusions}")
    if keywords:
        fulltext_parts.append(f"Keywords: {keywords}")
    
    fulltext = '\n'.join(fulltext_parts) if fulltext_parts else ""

    return {
        "abstract_code": code or None,
        "Title": title or None,
        "Authors": authors or None,
        "Affiliations": affiliations or None,
        "Introduction": intro or None,
        "Methods": methods or None,
        "Results": results or None,
        "Conclusions": conclusions or None,
        "Keywords": keywords or None,
        "Fulltext": fulltext or None,
    }


def split_persian_abstracts(text: str) -> List[str]:
    """Split a Persian abstract file into candidate abstracts."""
    # Use title patterns to split abstracts
    title_pattern = r'(?=^[A-Z][^:\n]*(?:study|investigation|analysis|evaluation|assessment|effect|impact|role|influence|distribution|using|meta-analysis)[^:\n]*$)'
    
    splits = re.split(title_pattern, text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Filter out small chunks and combine with next if needed
    abstracts = []
    current_abstract = ""
    
    for split in splits:
        split = split.strip()
        if not split:
            continue
            
        if len(split) < 100:  # Too small, probably part of previous
            current_abstract += "\n" + split
        else:
            if current_abstract:
                abstracts.append(current_abstract)
            current_abstract = split
    
    if current_abstract:
        abstracts.append(current_abstract)
    
    # Filter abstracts that have minimum content
    filtered_abstracts = []
    for abstract in abstracts:
        if (len(abstract) > 200 and 
            re.search(rf'(?i)(?:{INTRO_HDR}|{METHODS_HDR}|{RESULTS_HDR}|{CONCLUSION_HDR})', abstract)):
            filtered_abstracts.append(abstract)
    
    return filtered_abstracts


def split_into_abstract_blocks_corrected(full_text, file_name=""):
    """Split a file into abstract-sized blocks."""
    # Ensure consistent newlines
    full_text = full_text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Check if this is the Persian file
    if "persian" in file_name.lower() or "مقالات" in file_name:
        return split_persian_abstracts(full_text)
    
    # For other files, try to find P/O codes first
    matches = list(re.finditer(r'^\s*([PO]\d+)\b', full_text, flags=re.IGNORECASE | re.MULTILINE))
    
    if matches:
        # We have coded abstracts
        blocks = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            blocks.append(full_text[start:end].strip())
        return blocks
    
    # No codes found - use general splitting
    logger.info("No P/O codes found, using general splitting")
    return split_persian_abstracts(full_text)


def corrected_abstract_parser(file_path: str, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Parse one file and save the extracted abstracts."""
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []

    # Set output directory
    if output_dir:
        output_dir = Path(output_dir).resolve()
    else:
        output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract text from file
    logger.info(f"Processing file: {file_path}")
    full_text = extract_text_from_file(str(file_path))
    
    if not full_text.strip():
        logger.warning(f"No text extracted from {file_path}")
        return []

    # Normalize text
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    # Split into blocks
    blocks = split_into_abstract_blocks_corrected(full_text, file_path.name)
    logger.info(f"Found {len(blocks)} potential abstract blocks")

    # Parse each block
    parsed = []
    for i, b in enumerate(blocks):
        if not b.strip():
            continue
            
        # For non-coded abstracts, assign a sequential code
        block_code = ""
        if not re.search(r'^\s*[PO]\d+\b', b, re.IGNORECASE | re.MULTILINE):
            block_code = f"A{i+1}"  # A1, A2, A3 for non-coded abstracts
            
        d = parse_abstract_corrected(b, block_code)
        
        # Quality filter: keep if it has a code and at least title or one section
        if d["abstract_code"] and any([d["Title"], d["Authors"], d["Introduction"], d["Methods"], d["Results"], d["Conclusions"], d['Keywords']]):
            parsed.append(d)

    # Write JSON
    output_file = output_dir / f'corrected_parsed_abstracts_{file_path.stem}.json'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(parsed)} abstracts to {output_file}")
    except Exception as e:
        logger.error(f"Error saving to JSON: {e}")

    return parsed


def process_abs_data_folder_corrected(folder_path: str = "ABS_DATA", 
                                      output_dir: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Parse all supported abstract files in a folder."""
    folder_path = Path(folder_path).resolve()
    if not folder_path.exists():
        logger.error(f"Folder not found: {folder_path}")
        return {}

    results = {}
    supported_extensions = {'.docx', '.pdf', '.txt'}

    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            logger.info(f"Processing {file_path.name}")
            try:
                parsed_abstracts = corrected_abstract_parser(str(file_path), output_dir)
                results[file_path.name] = parsed_abstracts
                logger.info(f"Successfully processed {file_path.name}: {len(parsed_abstracts)} abstracts")
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                results[file_path.name] = []

    return results


def split_into_abstract_blocks_flexible(full_text):
    """Split text when abstract numbering is inconsistent."""
    # Ensure consistent newlines
    full_text = full_text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Strategy 1: Try P-codes first
    p_matches = list(re.finditer(r'^\s*(P\d+)\b', full_text, flags=re.IGNORECASE | re.MULTILINE))
    if p_matches:
        blocks = []
        for i, m in enumerate(p_matches):
            start = m.start()
            end = p_matches[i + 1].start() if i + 1 < len(p_matches) else len(full_text)
            blocks.append(full_text[start:end].strip())
        return blocks
    
    # Strategy 2: Split by other common abstract separators
    abstract_separators = [
        r'^\s*Abstract\s+\d+',  # "Abstract 1", "Abstract 2"
        r'^\s*\d+\.\s*[A-Z]',   # "1. Title", "2. Title"
        r'^\s*\d+\s*[:\-]\s*[A-Z]',  # "1: Title", "1- Title"
    ]
    
    for pattern in abstract_separators:
        matches = list(re.finditer(pattern, full_text, flags=re.IGNORECASE | re.MULTILINE))
        if len(matches) > 1:  # Need at least 2 matches to split
            blocks = []
            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
                blocks.append(full_text[start:end].strip())
            return blocks
    
    # Strategy 3: Split by multiple blank lines + title pattern
    title_pattern = r'\n\s*\n\s*\n+([A-Z][^\n]{10,100})\n'
    title_matches = list(re.finditer(title_pattern, full_text))
    if len(title_matches) > 1:
        blocks = []
        # Add first block (before first title match)
        if title_matches[0].start() > 0:
            blocks.append(full_text[:title_matches[0].start()].strip())
        
        for i, m in enumerate(title_matches):
            start = m.start()
            end = title_matches[i + 1].start() if i + 1 < len(title_matches) else len(full_text)
            blocks.append(full_text[start:end].strip())
        return [b for b in blocks if b.strip()]
    
    # Strategy 4: If no clear separators, return entire text as one block
    return [full_text]


def parse_abstract_block_flexible(block, code_number=None):
    """Parse a loosely formatted abstract block."""
    # Normalize spacing
    block = re.sub(r'[ \t]+', ' ', block)
    block = re.sub(r'\n{3,}', '\n\n', block)

    # 1) Code: Use provided code_number, or try to extract from text
    if code_number is not None:
        code = f"P{code_number}"
        # Remove P-code from beginning if it exists
        code_match = re.search(r'(?im)^\s*(P\d+)\b', block)
        if code_match:
            after_code = block[code_match.end():].lstrip('\n ')
        else:
            after_code = block
    else:
        # Try to extract existing code
        code_match = re.search(r'(?im)^\s*(P\d+)\b', block, flags=re.IGNORECASE)
        if code_match:
            code = code_match.group(1)
            after_code = block[code_match.end():].lstrip('\n ')
        else:
            # Try other numbering patterns
            other_code = re.search(r'(?im)^\s*((?:Abstract\s+\d+|\d+[\.\:\-]))', block)
            if other_code:
                code = other_code.group(1).strip()
                after_code = block[other_code.end():].lstrip('\n ')
            else:
                code = None
                after_code = block

    # 2) Title: first substantial line(s) that don't look like headers/affiliations
    title = ""
    stop_markers = re.compile(
        rf'^(?:{INTRO_HDR}|{METHODS_HDR}|{RESULTS_HDR}|{CONCLUSION_HDR}|{KEYWORDS_HDR})\s*:'
        r'|^\s*\d+\s*[:\-]'
        r'|^\s*(?:Department|Faculty|Professor|School)\b',
        flags=re.IGNORECASE | re.MULTILINE)

    # Split by paragraphs
    parts = re.split(r'\n\s*\n', after_code, maxsplit=1)
    if parts:
        candidate = parts[0].strip()
        # Remove any header markers
        sm = stop_markers.search(candidate)
        if sm:
            candidate = candidate[:sm.start()].strip()

        lines = [ln.strip() for ln in candidate.split('\n') if ln.strip()]
        if lines:
            title = " ".join(lines)

    # 3) Authors: similar logic to original
    authors = ""
    remainder = after_code[len(parts[0]):].lstrip('\n') if len(parts) > 1 else after_code
    
    m_auth = re.search(
        rf'^(.*?)\n\s*(?=(?:{INTRO_HDR}|{METHODS_HDR}|{RESULTS_HDR}|{CONCLUSION_HDR}|{KEYWORDS_HDR})\s*:'
        rf'|^\s*\d+\s*[:\-]|^\s*(?:Department|Faculty|Professor|School)\b|\Z)',
        remainder,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    if m_auth:
        authors = re.sub(r'\s*\n\s*', ' ', m_auth.group(1)).strip()

    # 4) Affiliations: same logic as original
    affiliations = ""
    aff_scan_start = m_auth.end() if m_auth else 0
    next_all = rf'{INTRO_HDR}|{METHODS_HDR}|{RESULTS_HDR}|{CONCLUSION_HDR}|{KEYWORDS_HDR}'
    next_hdr = re.search(rf'(?mi)^\s*(?:{next_all})\s*:?', remainder[aff_scan_start:])
    
    if next_hdr:
        aff_window = remainder[aff_scan_start: aff_scan_start + next_hdr.start()]
    else:
        aff_window = remainder[aff_scan_start:]

    AFFIL_LEAD = r'(?:\d+|[¹²³⁴⁵⁶⁷⁸⁹]+|[*†‡§¶#]+)[\)\]:\.\-]?'
    AFFIL_KEYS = (
        r'(?:Department|Faculty|School|Institute|Hospital|University|Center|Centre|Clinic|'
        r'College|Laboratory|Lab|Research\s+Center|Clinical\s+Microbiology\s+Research\s+Center)'
        r'\b'
    )
    AFFIL_LINE = rf'^\s*(?:{AFFIL_LEAD}\s+.+|{AFFIL_KEYS}.*)'

    m_aff_block = re.search(
        rf'(?mis){AFFIL_LINE}(?:\n+{AFFIL_LINE})*',
        aff_window,
        flags=re.MULTILINE
    )

    if m_aff_block:
        raw_aff = m_aff_block.group(0).strip()
        affiliations = re.sub(r'\s*\n\s*', ' ', raw_aff).strip()

    # 5) Sections: same as original
    next_all = rf'{INTRO_HDR}|{METHODS_HDR}|{RESULTS_HDR}|{CONCLUSION_HDR}|{KEYWORDS_HDR}'

    intro = _extract_section(remainder, INTRO_HDR, next_all)
    methods = _extract_section(remainder, METHODS_HDR, next_all)
    results = _extract_section(remainder, RESULTS_HDR, next_all)
    conclusions = _extract_section(remainder, CONCLUSION_HDR, next_all)

    # 6) Keywords
    kw = _extract_section(remainder, KEYWORDS_HDR, next_all)
    keywords = ", ".join([k.strip(" ,.;") for k in re.split(r'[;,]\s*|\s{2,}', kw) if k.strip()]) if kw else ""

    return {
        "abstract_code": code,
        "Fulltext": f"Introduction: {intro or None}\nMethods: {methods or None}\nResults: {results or None}\nConclusions: {conclusions or None}\nKeywords: {keywords or None}",
        "Title": title or None,
        "Authors": authors or None,
        "Affiliations": affiliations or None,
        "Introduction": intro or None,
        "Methods": methods or None,
        "Results": results or None,
        "Conclusions": conclusions or None,
        "Keywords": keywords or None,
    }


def abstract_parser_flexible(directory_path: str,
                           starting_code_number: int,
                           output_dir: Optional[str] = None):
    """Parse a folder of loosely formatted abstract files."""
    directory_path = Path(directory_path).resolve()
    
    if not directory_path.is_dir():
        raise ValueError(f"Provided path {directory_path} is not a directory")

    if output_dir:
        output_dir = Path(output_dir).resolve()
    else:
        output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_parsed = []
    processed_files = 0
    current_code_number = starting_code_number
    
    # Iterate through all files in the directory
    for file_path in directory_path.iterdir():
        if file_path.is_file():
            try:
                print(f"Processing file: {file_path.name}")
                
                full_text = extract_text_from_file(str(file_path))
                full_text = re.sub(r'\n{3,}', '\n\n', full_text)

                blocks = split_into_abstract_blocks_flexible(full_text)

                for b in blocks:
                    # Skip empty blocks
                    if not b.strip():
                        continue
                    
                    d = parse_abstract_block_flexible(b, current_code_number)
                    
                    # More flexible validation - accept if has title OR any section
                    if any([d["Title"], d["Introduction"], d["Methods"], d["Results"], d["Conclusions"], d['Keywords']]):
                        # Add source file information
                        d["source_file"] = file_path.name
                        all_parsed.append(d)
                        current_code_number += 1
                
                processed_files += 1
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                continue

    # Write JSON
    OUTPUT_JSON = output_dir / 'parsed_abstracts_flexible.json'
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_parsed, f, ensure_ascii=False, indent=2)

    print(f"Processed {processed_files} files and parsed {len(all_parsed)} abstracts to {OUTPUT_JSON}")
    print(f"Abstract codes range from P{starting_code_number} to P{current_code_number-1}")


if __name__ == "__main__":
    # Process all files
    results = process_abs_data_folder_corrected()
    total_abstracts = sum(len(abstracts) for abstracts in results.values())
    print(f"Processed {len(results)} files with {total_abstracts} total abstracts")
