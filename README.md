```markdown
# OGRRE Extract Studio

**OGRRE Extract Studio** is an interactive, web-based tool built with [Dash](https://dash.plotly.com/), [Plotly](https://plotly.com/), [Mantine](https://mantine.dev/), and [Docling](https://github.com/DS4SD/docling). It provides an end-to-end user interface and pipeline for parsing PDF documents, inspecting structural and textual annotations, modifying spatial extraction bounding boxes, and exporting structured JSON datasets and annotated PDFs.

---

## Key Features

- **Interactive PDF Viewer**: Multi-page PDF visualizer rendered on an interactive Plotly scatter plot canvas with zoom, bounding box hover overlays, and selection states.
- **Docling Extraction Engine**: Integrates with local extraction tools (via `DynamicExtractor`) to parse headers, paragraphs, lists, tables, pictures, and key-value pairs from complex documents.
- **Custom Post-Processing Workflows**:
  - **Key-Value Pair Merging**: Spatial- and pattern-aware matching to pair keys with nearby values.
  - **Boilerplate Removal**: Cross-page spatial and textual analysis to strip repetitive headers, footers, and page numbers.
  - **Table Structure Repair**: Intersecting block alignment and cell value repair using pre-computed row/column grid vectors.
  - **Zoned Block Merging**: Reading-order-aware merging of vertically aligned text lines.
- **Interactive Spatial Bounding Box Manipulation**: Split, merge, or ignore selected bounding boxes directly on the document canvas.
- **JSON Schema Configuration**: Input and apply custom JSON extraction schemas directly within the user interface modal.
- **Built-in Custom File Explorer**: File explorer modal with folder navigation, history tracking, extension filtering (`.pdf`, `.json`), and new file creation validation.
- **Conflict Management**: Integrated file overwrite warnings and queue management before saving outputs to disk.

---

## Architecture Overview

```
.
├── app.py                         # Main Dash UI application and callback orchestration
├── docling_postprocessing_gui_2.py # Document processing, AST filtering, and AST/table/boilerplate logic
├── extractor_integration_gui.py   # Wrapper for local extractor integration (DynamicExtractor)
├── file_explorer.py               # Utility logic for custom File Explorer component
└── utils_gui.py                   # Helper methods for PDF rendering, OCR, and coordinate geometry
└── assets                         # Static assets (images, CSS, and JS)
    └── images                     # Image assets for the UI
    └── styles.css                 # CSS styling for UI components
    └── workflow_scroll.js         # JavaScript logic for scrolling in workflow section
    └── zoom_handler.js            # JavaScript logic for zooming in the PDF viewer
```

---

## Installation & Setup

### Prerequisites

Ensure you have Python 3.9+ installed along with standard scientific and machine learning libraries (PyTorch/CUDA supported where applicable).
An environment manager like [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is recommended for managing Python dependencies.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CATALOG-Historic-Records/ogrre-extract.git
   cd ogrre-extract-studio
   ```

2. **Create and activate a Conda environment:**
   ```bash
   conda create -n ogrre-env python=3.10 -y
   conda activate ogrre-env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Hugging Face Authentication (Optional):**
   If using Hugging Face models for processing, place your Hugging Face API token in `hf_token.txt` at the root or parent directory level:
   ```bash
   echo "your_hf_token_here" > hf_token.txt
   ```

---

## Getting Started

Run the application locally:

```bash
python app.py
```

After starting the server, open your web browser and navigate to:
```
http://127.0.0.1:8050/
```

---

## User Workflow

1. **Select Input/Output Paths**:
   - Click **Browse** next to **PDF Input** to select a document.
   - Click **Browse** next to **JSON Output** (and optional **PDF Output**) to select the export targets.
2. **Configure Schema & Workflows**:
   - Toggle post-processing options under **Tables**, **Merge**, and **Boilerplate** accordions.
   - If desired, open **Configure Schema...** to set a targeted JSON output schema.
3. **Execute Extraction**:
   - Click **Run Docling** to extract document structures. A progress modal tracks background execution status.
4. **Inspect & Adjust Bounding Boxes**:
   - View structured overlays directly on the PDF viewer canvas.
   - Click or draw a selection box to select bounding boxes.
   - Use **Split Selected**, **Merge Selected**, or **Ignore Selected** to fine-tune layout regions.
5. **Preview & Save**:
   - Inspect the structured output in the live **JSON** side panel.
   - Click **Save** to export the structured JSON document (and annotated PDF). If target output files already exist, an overwrite prompt will request confirmation.

---

## Requirements & Core Dependencies

- [Dash](https://dash.plotly.com/) & [Dash Mantine Components](https://www.dash-mantine-components.com/)
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)
- [Plotly](https://plotly.com/python/)
- [Docling Core](https://github.com/DS4SD/docling-core)
- [OpenCV](https://opencv.org/) & [Pillow](https://python-pillow.org/)
- [Diskcache](https://grantjenks.com/docs/diskcache/)
```