import os
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(project_root / "cache")

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.utils import logging
logging.set_verbosity_error()
from transformers import RTDetrConfig


if "rt_detr_v2" not in CONFIG_MAPPING:
    CONFIG_MAPPING.register("rt_detr_v2", RTDetrConfig)

import io
import json
import dash
import pickle
import base64
import diskcache
from PIL import Image
from glob import glob
import pymupdf as fitz
from typing import Any
import plotly.graph_objects as go
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc

from .lib.file_explorer import get_directory_items, update_folder_history, get_file_list, check_file_creatable
from dash import (dcc, html, Input, Output, State, ctx, ALL, no_update, clientside_callback,
                  DiskcacheManager, MATCH)

# ---- DISKCACHE MANAGER ----
cache_dir = project_root / "cache_diskcache"
cache = diskcache.Cache(str(cache_dir))
background_callback_manager = DiskcacheManager(cache)

# ---- DOCLING EXTRACTOR ----
from ogrre_extract import DynamicExtractor
from ogrre_extract.lib.utils import get_page_dimensions

extractor_instance = None
hf_token_path = project_root / "hf_token.txt" if (project_root / "hf_token.txt").exists() else None
if hf_token_path:
    hf_token_path = str(hf_token_path)

def get_extractor():
    """Retrieves or initializes the singleton DynamicExtractor instance.

    Returns:
        DynamicExtractor: The active extractor instance for document processing.
    """
    global extractor_instance
    if extractor_instance is None:
        extractor_instance = DynamicExtractor(hf_token_path=hf_token_path)
    return extractor_instance


# ---- FILE EXPLORER ----
cwd = os.getcwd()
current_files = glob(cwd + '/*')
files_only = glob(cwd + '/*.*')
default_file = files_only[0] if len(files_only) > 0 else ''
current_header = ''
all_files = get_file_list(current_files, cwd, [])
current_tooltip = f'Current directory: {os.getcwd()}'

file_explorer_html = html.Div(
    id='gui-fileExplorerWindow',
    className='gui__fileExplorerWindow close',
    n_clicks=0,
    children=[
        html.Div(
            id='gui-fileExplorer-container',
            className='gui__fileExplorer--container',
            children=[
                html.Div(
                    id='gui-fileExplorer-header',
                    className='gui__fileExplorer--header',
                    children=[
                        html.Div(
                            id='gui-fileExplorer-header-title',
                            className='gui__fileExplorer--header__title',
                            children=[
                                html.H4(
                                    id='gui-fileExplorer-header-title-text',
                                    className='gui__fileExplorer--header__title--text',
                                    children=current_header
                                )
                            ]
                        ),
                        html.Div(
                            id='gui-fileExplorer-header-exit-button',
                            className='gui__fileExplorer--header__exitButton',
                            children=[],
                            n_clicks=0
                        )
                    ]
                ),
                html.Div(
                    id='gui-fileExplorer-directory-bar',
                    className='gui__fileExplorer--directoryBar',
                    children=[
                        html.Div(
                            id='gui-fileExplorer-directory-bar-backButton',
                            className='gui__fileExplorer--directoryBar__backButton--disabled',
                            children=[],
                            n_clicks=0,
                            disable_n_clicks=True,
                        ),
                        html.Div(
                            id='gui-fileExplorer-directory-bar-upButton',
                            className='gui__fileExplorer--directoryBar__upButton',
                            children=[],
                            n_clicks=0,
                            disable_n_clicks=False,
                        ),
                        html.Div(
                            id='gui-fileExplorer-directory-bar-cwd-container',
                            className='gui__fileExplorer--directoryBar__cwd--container',
                            children=[
                                html.Div(
                                    id='gui-fileExplorer-directory-bar-cwd',
                                    children=[dcc.Input(
                                        id='gui-fileExplorer-directory-bar-cwd-widget',
                                        className='genericInput__value',
                                        value=cwd,
                                        placeholder=cwd,
                                        spellCheck=False,
                                        debounce=True,
                                    )
                                    ]
                                )
                            ]
                        ),
                        html.Div(
                            id='gui-fileExplorer-directory-bar-error',
                            className='genericInput__error directoryBar__error close',
                            children=[
                                html.H4(
                                    id='gui-fileExplorer-directory-bar-error-text',
                                    className='genericInput__error--text',
                                    children=['']
                                )
                            ]
                        ),
                        dbc.Tooltip(
                            id='textInput-tooltip-fileExplorer-cwd',
                            className='genericInput__tooltip',
                            children=[current_tooltip],
                            target='gui-fileExplorer-directory-bar-cwd',
                            placement='bottom',
                            trigger='hover',
                            delay={'show': 250, 'hide': 100}
                        )
                    ]
                ),
                html.Div(
                    id='gui-fileExplorer-workspace',
                    className='gui__fileExplorer--workspace',
                    children=[
                        html.Div(
                            id='gui-fileExplorer-workspace-file-list',
                            className='gui__fileExplorer--workspace__fileList',
                            children=all_files
                        )
                    ]
                ),
                html.Div(
                    id='gui-fileExplorer-filename-bar',
                    className='gui__fileExplorer--filenameBar',
                    children=[
                        html.Div(
                            id='gui-fileExplorer-filename-bar-title',
                            className='gui__fileExplorer--filenameBar__title',
                            children=[
                                html.H4(
                                    id='gui-fileExplorer-filename-bar-title-text',
                                    className='gui__fileExplorer--filenameBar__title--text',
                                    children=['File Name:']
                                )
                            ]
                        ),
                        html.Div(
                            id='gui-fileExplorer-filename-bar-input',
                            className='gui__fileExplorer--filenameBar__input',
                            children=[
                                html.Div(
                                    id='gui-fileExplorer-filename-bar-input-container',
                                    className='gui__fileExplorer--filenameBar__input--container',
                                    children=[dcc.Input(
                                        id='gui-fileExplorer-filename',
                                        className='genericInput__value',
                                        value=current_files[0].split('\\')[-1] if current_files else '',
                                        placeholder=current_files[0].split('\\')[-1] if current_files else '',
                                        spellCheck=False,
                                        debounce=True,
                                    )
                                    ]
                                )
                            ]
                        ),
                        html.Div(
                            id='gui-fileExplorer-filename-bar-dropdown',
                            className='gui__fileExplorer--filenameBar__dropdown',
                            children=[
                                html.Div(
                                    id='gui-fileExplorer-filename-bar-dropdown-container',
                                    className='gui__fileExplorer--filenameBar__dropdown--container',
                                    children=[
                                        dcc.Dropdown(
                                            id='gui-fileExplorer-filename-bar-dropdown-widget',
                                            className=("dropdownInput__widget--dropdown "
                                                       "numericInput-withRange__type--widget"),
                                            options=[
                                                {'label': '.pdf', 'value': 'pdf'},
                                                {'label': '.json', 'value': 'json'},
                                                {'label': 'all files(*.*)', 'value': 'all'}
                                            ],
                                            value='all',
                                            persistence=True,
                                            searchable=False,
                                            clearable=False,
                                            multi=False,
                                            optionHeight=21
                                        )
                                    ]
                                )
                            ]
                        ),
                        html.Div(
                            id='gui-fileExplorer-filename-bar-error',
                            className='genericInput__error filenameBar__error close',
                            children=[
                                html.H4(
                                    id='gui-fileExplorer-filename-bar-error-text',
                                    className='genericInput__error--text',
                                    children=['']
                                )
                            ]
                        ),
                    ]
                ),
                html.Div(
                    id='gui-fileExplorer-buttons',
                    className='gui__fileExplorer--buttons',
                    children=[
                        html.Div(
                            id='gui-fileExplorer-buttons-action',
                            className='gui__fileExplorer--buttons__action',
                            children=[
                                html.Button(
                                    id='fileExplorer_actionButton',
                                    className='gui__fileExplorer--button',
                                    children=[
                                        html.H4(
                                            id='gui-fileExplorer-actionButton-text',
                                            className='gui__fileExplorer--actionButton__text',
                                            children=['Save']
                                        )
                                    ],
                                    n_clicks=0,
                                    value='saveFile',
                                )
                            ]
                        ),
                        html.Div(
                            id='gui-fileExplorer-buttons-cancel',
                            className='gui__fileExplorer--buttons__cancel',
                            children=[
                                html.Button(
                                    id='fileExplorer_cancelButton',
                                    className='gui__fileExplorer--button',
                                    children=[
                                        html.H4(
                                            id='gui-fileExplorer-cancelButton-text',
                                            className='gui__fileExplorer--cancelButton__text',
                                            children=['Cancel']
                                        )
                                    ],
                                    n_clicks=0,
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

# ---- PDF PAGE NAVIGATOR ----
pdf_nav_bar = html.Div(
    children=[
        dbc.Button(
            "◀",
            id="btn-prev-page",
            n_clicks=0,
            disabled=True,
            color="secondary",
            outline=True,
            size="sm",
            className="pdf-nav-btn pdf-nav-btn-prev",
        ),
        html.Div(
            className="pdf-nav-controls",
            children=[
                html.Span("Page ", className="pdf-nav-text"),
                dbc.Input(
                    id="input-page-number",
                    type="number",
                    min=1,
                    max=1,
                    value=1,
                    step=1,
                    size="sm",
                    className="pdf-nav-input",
                ),
                html.Span("of", className="pdf-nav-text"),
                html.Span("1", id="lbl-total-pages", className="pdf-nav-text"),
            ],
        ),
        dbc.Button(
            "▶",
            id="btn-next-page",
            n_clicks=0,
            disabled=True,
            color="secondary",
            outline=True,
            size="sm",
            className="pdf-nav-btn pdf-nav-btn-next",
        ),
    ],
)

# ---- SCHEMA MODAL ----
schema_modal = html.Div(
    id="gui-schemaModalWindow",
    className="gui__schemaModalWindow close",
    children=[
        html.Div(
            className="gui__schemaModal--container",
            children=[
                html.Div(
                    className="gui__schemaModal--header",
                    children=[
                        html.H4("Configure Extraction Schema", className="gui__schemaModal--title"),
                        html.Button(id="btn-close-schema-modal", className="gui__schemaModal--exitButton", n_clicks=0),
                    ]
                ),
                html.Div(
                    className="gui__schemaModal--body",
                    children=[
                        dcc.Textarea(
                            id="schema-text-modal",
                            placeholder="Enter JSON schema here...",
                            className="schema-modal-textarea"
                        )
                    ]
                ),
                html.Div(
                    className="gui__schemaModal--footer",
                    children=[
                        html.Button("Apply Schema", id="btn-apply-schema-modal", className="btn-action", n_clicks=0),
                        html.Button("Cancel", id="btn-cancel-schema-modal", className="btn-action btn-secondary", n_clicks=0),
                    ]
                )
            ]
        )
    ]
)

# ---- OVERWRITE CONFIRMATION MODAL ----
overwrite_modal = html.Div(
    id="gui-overwriteModalWindow",
    className="gui__overwriteModalWindow close",
    children=[
        html.Div(
            className="gui__overwriteModal--container",
            children=[
                html.Div(
                    className="gui__overwriteModal--header",
                    children=[
                        html.H4("File Already Exists", className="gui__overwriteModal--title"),
                        html.Button(id="btn-close-overwrite-modal", className="gui__schemaModal--exitButton", n_clicks=0),
                    ]
                ),
                html.Div(
                    className="gui__overwriteModal--body",
                    children=[
                        html.P(id="overwrite-modal-message", className="overwrite-modal-text")
                    ]
                ),
                html.Div(
                    className="gui__overwriteModal--footer",
                    children=[
                        html.Button("Cancel", id="btn-cancel-overwrite-modal", className="btn-action btn-secondary", n_clicks=0),
                        html.Button("Overwrite", id="btn-confirm-overwrite-modal", className="btn-action btn-danger", n_clicks=0),
                    ]
                )
            ]
        )
    ]
)

# ---- PROGRESS MODAL ----
progress_modal = html.Div(
    id="gui-progressModalWindow",
    className="gui__progressModalWindow close",
    children=[
        html.Div(
            className="gui__progressModal--container",
            children=[
                html.Div(
                    className="gui__progressModal--header",
                    children=[
                        html.H4("Processing Document", className="gui__progressModal--title"),
                    ],
                ),
                html.Div(
                    className="gui__progressModal--body",
                    children=[
                        dbc.Progress(
                            id="docling-progress-bar",
                            value=0,
                            min=0,
                            max=100,
                            animated=True,
                            striped=True,
                            className="gui__progressModal--bar",
                        ),
                        html.Div(
                            id="docling-status-text",
                            children="Initializing...",
                            className="gui__progressModal--status",
                        ),
                    ],
                ),
            ],
        )
    ],
)

# ---- RAW EXPORT WARNING MODAL ----
raw_export_warning_modal = html.Div(
    id="raw-export-warning-modal",
    className="gui__rawWarningModalWindow close",
    children=[
        html.Div(
            className="gui__rawWarningModal--container",
            children=[
                html.Div(
                    className="gui__rawWarningModal--header",
                    children=[
                        html.H4("Raw JSON Export Warning", className="gui__rawWarningModal--title"),
                        html.Button(id="btn-close-raw-warning-x", className="gui__rawWarningModal--exitButton"),
                    ],
                ),
                html.Div(
                    className="gui__rawWarningModal--body",
                    children=[
                        html.P(
                            "Warning: The export will be saved in a minimally-processed JSON format and will require further postprocessing to be used in OGRRE.",
                            className="raw-warning-modal-text",
                        ),
                        html.P("Do you want to continue?", className="raw-warning-modal-prompt"),
                    ],
                ),
                html.Div(
                    className="gui__rawWarningModal--footer",
                    children=[
                        html.Button("Cancel", id="btn-cancel-raw-export", className="btn-secondary"),
                        html.Button("Continue", id="btn-confirm-raw-export", className="btn-action"),
                    ],
                ),
            ],
        ),
    ],
)

# ---- RAW EXPORT COMPLETE MODAL ----
raw_export_complete_modal = html.Div(
    id="raw-export-complete-modal",
    className="gui__rawCompleteModalWindow close",
    children=[
        html.Div(
            className="gui__rawCompleteModal--container",
            children=[
                html.Div(
                    className="gui__rawCompleteModal--header",
                    children=[
                        html.H4("Export Complete", className="gui__rawCompleteModal--title"),
                        html.Button(id="btn-close-raw-complete-x", className="gui__rawCompleteModal--exitButton"),
                    ],
                ),
                html.Div(
                    className="gui__rawCompleteModal--body",
                    children=[
                        html.P("Raw Export Complete!", className="raw-complete-modal-status"),
                        html.P(
                            "The unprocessed JSON file has been successfully exported to your selected file path.",
                            className="raw-complete-modal-text",
                        ),
                    ],
                ),
                html.Div(
                    className="gui__rawCompleteModal--footer",
                    children=[
                        html.Button("Continue", id="btn-confirm-raw-complete", className="btn-action"),
                    ],
                ),
            ],
        ),
    ],
)

# ---- DASH APP LAYOUT AND INITIALIZATION ----
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP,
                          "https://unpkg.com/@mantine/core@7/styles.css"],
    background_callback_manager=background_callback_manager,
    suppress_callback_exceptions=True
)

workflow_options = {
    "text": [
        {"label": " Re-OCR Low Confidence Text", "value": "reOCR_text", "disabled": False},
        {"label": " Spell Check", "value": "spell_check", "disabled": False}
    ],
    "tables": [
        {"label": " Fix Tables", "value": "fix_tables", "disabled": False}
    ],
    "merge": [
        {"label": " Merge Key-Value Pairs", "value": "merge_kv", "disabled": False},
        {"label": " Auto-merge Proximity", "value": "auto_merge", "disabled": False}
    ],
    "boilerplate": [
        {"label": " Detect Boilerplate", "value": "detect_boilerplate", "disabled": False},
        {"label": " Remove Excess", "value": "remove_boilerplate", "disabled": False}
    ],
}

app.layout = dmc.MantineProvider(
    children=html.Div(
        className="app-container",
        children=[
            # Stores for state management
            dcc.Store(id="doc-store", data={}),
            dcc.Store(id="explorer-config-store", data=[]),
            dcc.Store(id="explorer-state-store", data=None),
            dcc.Store(id='file-explorer-trigger', data=None),
            dcc.Store(id='previous-folder-locations', data={'dirs': [os.getcwd()]}),
            dcc.Store(id="schema-text", data=""),
            dcc.Store(id="overwrite-queue-store", data={"queue": [], "confirmed": []}),
            dcc.Store(id="selected-boxes-store", data=[]),
            dcc.Store(id="file-explorer-double-click-trigger", data=0),

            # Backdrop & Modal
            html.Div(id='gui-backdrop', className='gui__backdrop close'),
            file_explorer_html,
            schema_modal,
            overwrite_modal,
            progress_modal,
            raw_export_warning_modal,
            raw_export_complete_modal,

            # Header
            html.Div(
                className="header-container",
                children=[
                    html.Div(
                        className="logo-box",
                        children=[
                            html.Img(
                                src="assets/images/ogrre-logo.png",
                                alt="OGRRE Logo",
                                className="logo-img",
                            )
                        ],
                    ),
                    html.Div("OGRRE EXTRACT STUDIO", className="title-box"),
                ],
            ),

            # Path Control Bar
            html.Div(
                className="path-control-bar",
                children=[
                    html.Div(
                        className="path-group",
                        children=[
                            html.Label("PDF Input:"),
                            dcc.Input(id="pdf-input-path", type="text", className="path-input"),
                            html.Button(
                                "Browse",
                                id="btn-browse-pdf-input",
                                className="btn-action",
                                style={"marginRight": "10px"},
                            ),
                        ]
                    ),
                    html.Div(
                        className="path-group",
                        children=[
                            html.Label("JSON Output:"),
                            dcc.Input(id="json-output-path", type="text", className="path-input"),
                            html.Button(
                                "Browse",
                                id="btn-browse-json-output",
                                className="btn-action",
                                style={"marginRight": "10px"},
                            ),
                        ]
                    ),
                    html.Div(
                        className="path-group",
                        children=[
                            dcc.Checklist(
                                id="export-pdf-check",
                                options=[{"label": " PDF Output:", "value": "export_pdf"}],
                                value=["export_pdf"],
                            ),
                            dcc.Input(id="pdf-output-path", type="text", className="path-input", disabled=False),
                            html.Button(
                                "Browse", id="btn-browse-pdf-output", className="btn-action", disabled=False,
                            ),
                        ]
                    ),
                ],
            ),

            # Main 3-Column Layout
            html.Div(
                className="main-grid",
                children=[
                    # Left Column: Workflow
                    html.Div(
                        className="panel-box",
                        children=[
                            html.Div("Workflow", className="section-title full-width"),
                            html.Div(className="workflow-container", children=[
                                html.Div(
                                    className="workflow-column",
                                    children=[
                                        html.Div("Postprocessing Options",
                                                 className="sub-heading workflow__header"),
                                        dmc.Accordion(
                                            id="workflow-options-accordion",
                                            chevronPosition="right",
                                            children=[
                                                dmc.AccordionItem(
                                                    value="text",
                                                    children=[
                                                        dmc.AccordionControl("Text", id="accordion-control-text"),
                                                        dmc.AccordionPanel(
                                                            dcc.Checklist(
                                                                id="workflow-options-text",
                                                                options=workflow_options["text"],
                                                                value=[wf_opt["value"] for wf_opt
                                                                       in workflow_options["text"]],
                                                                className="workflow-checklist",
                                                            )
                                                        ),
                                                    ],
                                                ),
                                                dmc.AccordionItem(
                                                    value="tables",
                                                    children=[
                                                        dmc.AccordionControl("Tables", id="accordion-control-tables"),
                                                        dmc.AccordionPanel(
                                                            dcc.Checklist(
                                                                id="workflow-options-tables",
                                                                options=workflow_options["tables"],
                                                                value=[wf_opt["value"] for wf_opt
                                                                       in workflow_options["tables"]],
                                                                className="workflow-checklist",
                                                            )
                                                        ),
                                                    ],
                                                ),
                                                dmc.AccordionItem(
                                                    value="merge",
                                                    children=[
                                                        dmc.AccordionControl("Merge", id="accordion-control-merge"),
                                                        dmc.AccordionPanel(
                                                            dcc.Checklist(
                                                                id="workflow-options-merge",
                                                                options=workflow_options["merge"],
                                                                value=[wf_opt["value"] for wf_opt
                                                                       in workflow_options["merge"]],
                                                                className="workflow-checklist",
                                                            )
                                                        ),
                                                    ],
                                                ),
                                                dmc.AccordionItem(
                                                    value="boilerplate",
                                                    children=[
                                                        dmc.AccordionControl("Boilerplate",
                                                                             id="accordion-control-boilerplate"),
                                                        dmc.AccordionPanel(
                                                            dcc.Checklist(
                                                                id="workflow-options-boilerplate",
                                                                options=workflow_options['boilerplate'],
                                                                value=[wf_opt["value"] for wf_opt
                                                                       in workflow_options["boilerplate"]],
                                                                className="workflow-checklist",
                                                            )
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="workflow-column",
                                    children=[
                                        html.Div(
                                            className="workflow__run--options",
                                            children=[
                                                html.Button("Configure Schema...", id="btn-open-schema-modal",
                                                            className="btn-action btn-full", style={"marginTop": "6px"}),
                                                html.Button("Export Raw JSON", id="btn-export-raw", className="btn-action btn-full", disabled=True),
                                                html.Button("Run Extractor", id="btn-run-docling", className="btn-action btn-full", disabled=True),
                                            ]
                                        ),
                                    ]
                                ),
                                html.Div(
                                    className="workflow-column",
                                    children=[
                                        html.Div(
                                            className="workflow__selection--options",
                                            children=[
                                                html.Div("Selection:", className="sub-heading"),
                                                html.Button("Split", id="btn-split", className="btn-action btn-third"),
                                                html.Button("Merge", id="btn-merge", className="btn-action btn-third"),
                                                html.Button("Ignore", id="btn-ignore", className="btn-action btn-third"),
                                            ]
                                        )
                                    ]
                                ),
                            ]),
                        ],
                    ),
                    # Center Column: Viewer
                    html.Div(
                        className="panel-box",
                        children=[
                            html.Div(
                                className="pdf-header-container",
                                children=[
                                    html.Div("PDF", className="section-title"),
                                    html.Div(
                                        id='pdf-layer-dropdown-wrapper',
                                        className='pdf-layer-dropdown-wrapper',
                                        children=[
                                            html.Button(
                                                "Layers ▾",
                                                id="btn-toggle-layer-dropdown",
                                                className="btn-action btn-layer-toggle",
                                                n_clicks=0,
                                            ),
                                            html.Div(
                                                id="pdf-layer-dropdown-menu",
                                                className="pdf-layer-dropdown-menu close",
                                                children=[
                                                    html.Div("Filter Layers",
                                                             className="pdf-layer-dropdown-title"
                                                    ),
                                                    dcc.Checklist(
                                                        id="block-type-filter",
                                                        options=[],
                                                        value=[],
                                                        inline=False,
                                                        className="block-type-checklist",
                                                    )
                                                ]
                                            )
                                        ]
                                    ),
                                    html.Div(id="pdf-nav-bar", className="pdf-nav-bar", children=pdf_nav_bar)
                                ]
                            ),
                            html.Div(
                                id="pdf-viewer-container",
                                className="pdf-viewer-container",
                                children=[
                                    html.Button(id="blank-click-trigger", style={"display": "none"}, n_clicks=0),
                                    dcc.Graph(
                                        id="pdf-graph",
                                        className="pdf-graph-container",
                                        responsive=True,
                                        config={
                                            "modeBarButtonsToAdd": ["drawrect", "eraseshape"],
                                            "displayModeBar": True,
                                            "responsive": True,
                                            "scrollZoom": False,
                                        },
                                    )
                                ],
                            )
                        ],
                    ),
                    # Right Column: JSON
                    html.Div(
                        className="panel-box",
                        children=[
                            html.Div("JSON", className="section-title"),
                            dmc.ScrollArea(
                                id="json-preview-container",
                                className="json-preview-scroll-area",
                                children=[
                                    html.Div(id="json-tree-view", children=[])
                                ],
                                style={"height": "calc(100% - 80px)"},
                            ),
                            html.Div(
                                className="save-container",
                                children=[
                                    html.Button("Save JSON", id="btn-save", className="btn-action btn-save"),
                                    html.Button("Save JSON & Metadata",
                                                id="btn-save-metadata",
                                                className="btn-action btn-save"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    ),
)


# --- CALLBACKS ---

# Combined PDF Loading & Page Navigation Handler
@app.callback(
    Output("doc-store", "data"),
    Output("input-page-number", "value"),
    Output("input-page-number", "max"),
    Output("lbl-total-pages", "children"),
    Output("btn-prev-page", "disabled"),
    Output("btn-next-page", "disabled"),
    Output("pdf-nav-bar", "className"),
    Input("pdf-input-path", "value"),
    Input("btn-prev-page", "n_clicks"),
    Input("btn-next-page", "n_clicks"),
    Input("input-page-number", "value"),
    State("doc-store", "data"),
    prevent_initial_call=True,
)
def manage_pdf_and_navigation(pdf_path, prev_clicks, next_clicks, typed_page_val, doc_data):
    """Manages initial PDF loading, state store initialization, and multi-page navigation controls.

    Args:
        pdf_path (str): File path to the target input PDF document.
        prev_clicks (int): Number of clicks on the previous page button.
        next_clicks (int): Number of clicks on the next page button.
        typed_page_val (int): Numeric page number typed into the page input box.
        doc_data (dict): Current document state payload stored in `doc-store`.

    Returns:
        tuple: A 7-element tuple updating document data, current page value, maximum page limit,
            total page count label, previous button state, next button state, and nav bar visibility class.
    """
    triggered_id = ctx.triggered_id

    # Load New PDF
    if triggered_id == "pdf-input-path":
        if not pdf_path or not os.path.isfile(pdf_path) or not pdf_path.lower().endswith(".pdf"):
            return [dash.no_update] * 7

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            page = doc[0]
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_height, page_width, scale_x, scale_y = get_page_dimensions(page, pix)

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            encoded_img = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
            doc.close()

            new_doc_data = {
                "pdf_path": pdf_path,
                "image_str": encoded_img,
                "width": pix.width,
                "height": pix.height,
                "page_width": page_width,
                "page_height": page_height,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "blocks": [],
                "document_json": {},
                "document_metadata": {},
                "total_pages": total_pages,
                "current_page": 0,
            }

            return (
                new_doc_data,
                1,
                total_pages,
                str(total_pages),
                True,
                total_pages < 2,
                "pdf-nav-bar visible",
            )
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return [dash.no_update] * 7

    # Page Navigation
    if not doc_data or "total_pages" not in doc_data:
        return [dash.no_update] * 7

    total_pages = doc_data.get("total_pages", 1)
    current_idx = doc_data.get("current_page", 0)
    current_num = current_idx + 1

    target_num = current_num

    if triggered_id == "btn-prev-page":
        target_num = max(1, current_num - 1)
    elif triggered_id == "btn-next-page":
        target_num = min(total_pages, current_num + 1)
    elif triggered_id == "input-page-number":
        if typed_page_val is None or typed_page_val == current_num:
            return [dash.no_update] * 7
        target_num = max(1, min(total_pages, int(typed_page_val)))

    target_idx = target_num - 1
    page_cache_key = f"{doc_data['pdf_path']}_page_{target_idx}"

    if page_cache_key in cache:
        page_info = cache[page_cache_key]
        doc_data["current_page"] = target_idx
        doc_data["image_str"] = page_info["image_str"]
        doc_data["width"] = page_info["width"]
        doc_data["height"] = page_info["height"]
        doc_data["page_width"] = page_info["page_width"]
        doc_data["page_height"] = page_info["page_height"]
        doc_data["scale_x"] = page_info["scale_x"]
        doc_data["scale_y"] = page_info["scale_y"]
        doc_data["blocks"] = page_info["blocks"]
    else:
        try:
            doc = fitz.open(doc_data["pdf_path"])
            page = doc[target_idx]
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_height, page_width, scale_x, scale_y = get_page_dimensions(page, pix)

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            encoded_img = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
            doc.close()

            doc_data["current_page"] = target_idx
            doc_data["image_str"] = encoded_img
            doc_data["width"] = pix.width
            doc_data["height"] = pix.height
            doc_data["page_width"] = page_width
            doc_data["page_height"] = page_height
            doc_data["scale_x"] = scale_x
            doc_data["scale_y"] = scale_y
            doc_data["blocks"] = []
        except Exception as e:
            print(f"Error rendering PDF page {target_idx}: {e}")

    prev_disabled = target_num <= 1
    next_disabled = target_num >= total_pages

    return (
        doc_data,
        target_num,
        dash.no_update,
        dash.no_update,
        prev_disabled,
        next_disabled,
        dash.no_update,
    )


# Raw Export Warning Modal Toggle
@app.callback(
    Output("raw-export-warning-modal", "className"),
    Output("gui-backdrop", "className", allow_duplicate=True),
    Input("btn-export-raw", "n_clicks"),
    Input("btn-close-raw-warning-x", "n_clicks"),
    Input("btn-cancel-raw-export", "n_clicks"),
    Input("btn-confirm-raw-export", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_raw_export_warning(btn_open, btn_x, btn_cancel, btn_confirm):
    """Toggles visibility of the raw export warning modal and UI backdrop overlay.

    Args:
        btn_open (int): Click count for the open raw export warning button.
        btn_x (int): Click count for the modal close button.
        btn_cancel (int): Click count for the modal cancel button.
        btn_confirm (int): Click count for the modal confirmation button.

    Returns:
        tuple: CSS class names updating the warning modal and global backdrop states.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "btn-export-raw":
        # Show warning modal and open the backdrop
        return "gui__rawWarningModalWindow open", "gui__backdrop open"

    if trigger_id == "btn-confirm-raw-export":
        # Close warning modal, but keep backdrop OPEN because progress modal will take over
        return "gui__rawWarningModalWindow close", "gui__backdrop open"

    # For Cancel or X, close both modal and backdrop
    return "gui__rawWarningModalWindow close", "gui__backdrop close"


# Raw Export Complete Modal Closing Callback
@app.callback(
    Output("raw-export-complete-modal", "className", allow_duplicate=True),
    Output("gui-backdrop", "className", allow_duplicate=True),
    Input("btn-close-raw-complete-x", "n_clicks"),
    Input("btn-confirm-raw-complete", "n_clicks"),
    prevent_initial_call=True,
)
def close_raw_export_complete(btn_x, btn_confirm):
    """Closes the raw export complete modal and resets the UI backdrop.

    Args:
        btn_x (int): Click count for the exit icon button.
        btn_confirm (int): Click count for the modal continue button.

    Returns:
        tuple: CSS class names closing the modal and background backdrop.
    """
    return "gui__rawCompleteModalWindow close", "gui__backdrop close"


# Combined Extraction & Raw Export Runner
@app.callback(
    Output("doc-store", "data", allow_duplicate=True),
    Output("raw-export-complete-modal", "className"),
    Output("gui-backdrop", "className", allow_duplicate=True),
    Input("btn-run-docling", "n_clicks"),
    Input("btn-confirm-raw-export", "n_clicks"),
    State("doc-store", "data"),
    State("workflow-options-text", "value"),
    State("workflow-options-tables", "value"),
    State("workflow-options-merge", "value"),
    State("workflow-options-boilerplate", "value"),
    State("json-output-path", "value"),
    State("schema-text", "data"),
    background=True,
    running=[
        (Output("btn-run-docling", "disabled"), True, False),
        (Output("btn-run-docling", "children"), "Processing...", "Run Extractor"),
        (Output("btn-export-raw", "disabled"), True, False),
        (Output("btn-export-raw", "children"), "Exporting...", "Export Raw JSON"),
        (Output("gui-progressModalWindow", "className"), "gui__progressModalWindow open", "gui__progressModalWindow close"),
        (Output("gui-backdrop", "className", allow_duplicate=True), "gui__backdrop open", "gui__backdrop close"),
    ],
    progress=[
        Output("docling-progress-bar", "value"),
        Output("docling-status-text", "children"),
    ],
    progress_default=(0, "Initializing..."),
    prevent_initial_call=True,
)
def run_docling(
    set_progress,
    n_clicks_docling,
    n_clicks_raw_confirm,
    doc_data,
    text_options,
    table_options,
    merge_options,
    boilerplate_options,
    json_output_path,
    schema_str,
):
    """Runs extraction workflows using Docling with local disk caching and progress updates.

    Args:
        set_progress (function): Dash background callback progress updater.
        n_clicks_docling (int): Click count for the standard extraction trigger button.
        n_clicks_raw_confirm (int): Click count for the raw export confirmation button.
        doc_data (dict): Current document metadata and page properties stored in `doc-store`.
        text_options (list): Selected workflow options for text postprocessing.
        table_options (list): Selected workflow options for table postprocessing.
        merge_options (list): Selected workflow options for key-value merging.
        boilerplate_options (list): Selected workflow options for boilerplate removal.
        json_output_path (str): File path specified for saving JSON output.
        schema_str (str): Active schema JSON configuration string.

    Returns:
        tuple: Updated document state dictionary, export completion modal CSS class, and backdrop CSS class.
    """
    if not doc_data or "pdf_path" not in doc_data:
        return dash.no_update, "gui__rawCompleteModalWindow close", "gui__backdrop close"

    # Identify trigger source
    triggered_id = ctx.triggered_id
    is_raw_export = (triggered_id == "btn-confirm-raw-export")

    try:
        cache.evict()
    except Exception:
        pass

    set_progress((5, "Checking cache..."))
    time.sleep(0.05)

    # Determine if a valid non-empty schema string is available
    pdf_path = doc_data["pdf_path"]

    # Define cache paths based on the input PDF filename
    cache_dir = project_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_filename = Path(pdf_path).name
    base_cache_name = Path(pdf_filename).stem
    cache_pdf_path = cache_dir / f"{base_cache_name}_annotated.pdf"
    cache_data_path = cache_dir / f"{base_cache_name}_extracted_data.pkl"

    annotated_doc = None
    blocks_by_page = None
    document_json_by_page = None
    document_metadata_by_page = None

    # Attempt to load from cache if this is a standard run (not raw export)
    if not is_raw_export and cache_pdf_path.exists() and cache_data_path.exists():
        try:
            set_progress((20, "Loading cached extraction results..."))
            time.sleep(2)

            # Load annotated document as a PyMuPDF document
            annotated_doc = fitz.open(str(cache_pdf_path))

            # Load dictionary structures via pickle
            with open(cache_data_path, "rb") as f:
                cached_payload = pickle.load(f)
                blocks_by_page = cached_payload.get("blocks_by_page")
                document_json_by_page = cached_payload.get("document_json_by_page")
                document_metadata_by_page = cached_payload.get("document_metadata_by_page")

            # Validate that all 4 required variables are present and valid
            if not (annotated_doc and blocks_by_page is not None and
                    document_json_by_page is not None and document_metadata_by_page is not None):
                annotated_doc = None
                blocks_by_page = None
                document_json_by_page = None
                document_metadata_by_page = None
            else:
                set_progress((50, "Cache loaded successfully! Skipping extraction."))
                time.sleep(0.05)
        except Exception as e:
            print(f"Cache load failed ({e}), falling back to live extraction...")
            annotated_doc = None
            blocks_by_page = None
            document_json_by_page = None
            document_metadata_by_page = None

    # Fall back to running live extraction if cache wasn't used or loaded
    if annotated_doc is None or blocks_by_page is None or document_json_by_page is None or document_metadata_by_page is None:
        set_progress((10, "Initializing Extractor and VLM Models...\nIf this is your first time, this may take several "
                          "minutes.\nSee console/terminal for details."))
        time.sleep(0.05)
        extractor = get_extractor()
        active_schema = schema_str.strip() if (schema_str and isinstance(schema_str, str) and schema_str.strip()) else None

        if is_raw_export:
            set_progress((20, "Running Raw Extraction (Bypassing Postprocessing)..."))
            workflow_opts = None
        else:
            workflow_opts = (text_options or []) + (table_options or []) + (merge_options or []) + (boilerplate_options or [])

        time.sleep(0.05)

        def gui_progress_bridge(pct, msg):
            set_progress((pct, msg))
            time.sleep(0.02)

        annotated_doc, blocks_by_page, document_json_by_page, document_metadata_by_page = extractor.extract_from_file(
            pdf_path,
            json_output_path,
            domain_dictionary="domain_words.txt",
            schema=active_schema,
            workflow_options=workflow_opts,
            progress_callback=gui_progress_bridge,
        )

        # Save successful standard extractions to cache for future development
        if not is_raw_export:
            try:
                annotated_doc.save(str(cache_pdf_path))
                with open(cache_data_path, "wb") as f:
                    pickle.dump({
                        "blocks_by_page": blocks_by_page,
                        "document_json_by_page": document_json_by_page,
                        "document_metadata_by_page": document_metadata_by_page
                    }, f)
                print(f"Successfully cached extraction output for {pdf_filename}.")
            except Exception as e:
                print(f"Failed to save extraction cache: {e}")

    set_progress((65, "Processing Rendered Page Views..."))
    time.sleep(0.05)

    total_pages = len(annotated_doc)

    for page_num in range(total_pages):
        progress_pct = int(65 + (30 * (page_num + 1) / total_pages))
        set_progress((progress_pct, f"Rendering Page {page_num + 1} of {total_pages}..."))
        time.sleep(0.05)

        page = annotated_doc[page_num]
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_height, page_width, scale_x, scale_y = get_page_dimensions(page, pix)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded_img = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

        blocks = blocks_by_page.get(page_num, [])

        page_payload = {
            "image_str": encoded_img,
            "width": pix.width,
            "height": pix.height,
            "page_width": page_width,
            "page_height": page_height,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "blocks": blocks,
        }
        cache[f"{pdf_path}_page_{page_num}"] = page_payload

    p0 = cache.get(f"{pdf_path}_page_0", {})

    final_doc_data = {
        **doc_data,
        "document_json_by_page": document_json_by_page,
        "document_metadata_by_page": document_metadata_by_page,
        "total_pages": total_pages,
        "current_page": 0,
        "image_str": p0.get("image_str", ""),
        "width": p0.get("width", 0),
        "height": p0.get("height", 0),
        "blocks": p0.get("blocks", []),
    }

    set_progress((100, "Extraction Complete!"))
    time.sleep(0.05)

    # Determine completion state modal classes
    if is_raw_export:
        return final_doc_data, "gui__rawCompleteModalWindow open", "gui__backdrop open"
    else:
        return final_doc_data, "gui__rawCompleteModalWindow close", "gui__backdrop close"


# Canvas Block Modification Actions (Split / Merge / Ignore)
@app.callback(
    Output("doc-store", "data", allow_duplicate=True),
    Output("selected-boxes-store", "data", allow_duplicate=True),
    Input("btn-split", "n_clicks"),
    Input("btn-merge", "n_clicks"),
    Input("btn-ignore", "n_clicks"),
    State("pdf-graph", "selectedData"),
    State("selected-boxes-store", "data"),
    State("doc-store", "data"),
    prevent_initial_call=True,
)
def handle_selection_actions(split_click, merge_click, ignore_click, selected_data,
                             click_selected_ids, doc_data):
    """Handles canvas block operations including splitting, merging, or ignoring bounding boxes.

    Args:
        split_click (int): Click count for the split action button.
        merge_click (int): Click count for the merge action button.
        ignore_click (int): Click count for the ignore action button.
        selected_data (dict): Bounding box selection range data from Plotly graph drag interactions.
        click_selected_ids (list): IDs of currently selected bounding boxes.
        doc_data (dict): Document payload stored in `doc-store`.

    Returns:
        tuple: Updated document payload dictionary and cleared selection IDs list.
    """
    if not doc_data or "blocks" not in doc_data:
        return no_update, no_update

    triggered = ctx.triggered_id
    blocks = doc_data["blocks"]
    selected_ids = set(click_selected_ids or [])

    if selected_data and "range" in selected_data:
        x_range = selected_data["range"]["x"]
        y_range = selected_data["range"]["y"]

        drag_ids = [
            b["block_num"] for b in blocks
            if (min(x_range) <= b["coords"][2] and max(x_range) >= b["coords"][0]) and
               (min(y_range) <= b["coords"][3] and max(y_range) >= b["coords"][1])
        ]
        selected_ids.update(drag_ids)

    selected_ids = list(selected_ids)
    if not selected_ids:
        return no_update, no_update

    if triggered == "btn-ignore":
        doc_data["blocks"] = [b for b in blocks if b["block_num"] not in selected_ids]

    elif triggered == "btn-merge" and len(selected_ids) > 1:
        merged_target = [b for b in blocks if b["block_num"] in selected_ids]
        new_x0 = min(b["coords"][0] for b in merged_target)
        new_y0 = min(b["coords"][1] for b in merged_target)
        new_x1 = max(b["coords"][2] for b in merged_target)
        new_y1 = max(b["coords"][3] for b in merged_target)
        merged_val = " ".join(b.get("value") or b.get("text", "") for b in merged_target)
        merged_confidences = [b.get("confidence") for b in merged_target if b.get("confidence") is not None]
        merged_type = [b.get("type") for b in merged_target]

        new_blocks = [b for b in blocks if b["block_num"] not in selected_ids]
        new_blocks.append({
            "block_num": max([b["block_num"] for b in blocks] + [0]) + 1,
            "key": merged_target[0].get("key", "Merged_Block"),
            "value": merged_val,
            "coords": [new_x0, new_y0, new_x1, new_y1],
            "type": "table_text" if "table_text" in merged_type
                    else ("key_value_pair" if "key_value_pair" in merged_type else "text"),
            "confidence": (sum(merged_confidences) / len(merged_confidences)) if merged_confidences else None,
            "original_blocks": merged_target,
        })
        doc_data["blocks"] = new_blocks

    return doc_data, []


# Dynamic Block Type Filter Populator Callback
@app.callback(
    Output("block-type-filter", "options"),
    Output("block-type-filter", "value"),
    Output("pdf-layer-dropdown-wrapper", "className"),
    Input("doc-store", "data"),
    State("block-type-filter", "value"),
)
def update_block_type_filter(doc_data, current_selected):
    """Populates block type filter options dynamically based on active document blocks.

    Args:
        doc_data (dict): Document payload stored in `doc-store`.
        current_selected (list): List of currently selected block type strings.

    Returns:
        tuple: Checklist options formatted with HTML labels, selected values list, and wrapper class.
    """
    types_set = set()
    if doc_data:
        for b in doc_data.get("blocks", []):
            t = b.get("type")
            if t:
                types_set.add(str(t))
        for p_info in doc_data.get("pages", {}).values():
            for b in p_info.get("blocks", []):
                t = b.get("type")
                if t:
                    types_set.add(str(t))

    if not types_set:
        return [], [], "pdf-layer-dropdown-wrapper close"

    sorted_types = sorted(list(types_set))

    options = []
    for t in sorted_types:
        label_text = t.replace("_", " ").title()
        color_code = get_block_type_color(t)

        options.append({
            "label": html.Span(
                label_text,
                style={"color": color_code},
            ),
            "value": t
        })

    current_selected = current_selected or []
    new_value = [t for t in sorted_types if t in current_selected]
    if not new_value:
        new_value = sorted_types

    return options, new_value, "pdf-layer-dropdown-wrapper"


# Display & Preview Renderer
@app.callback(
    Output("pdf-graph", "figure"),
    Output("json-tree-view", "children"),
    Output("pdf-graph", "style"),
    Output("pdf-viewer-container", "className"),
    Input("doc-store", "data"),
    State("selected-boxes-store", "data"),
    State("pdf-viewer-container", "className"),
)
def update_display(doc_data, selected_box_ids, pdf_viewer_className):
    """Renders the viewer figure, overlays bounding box traces, and generates the JSON tree view.

    Args:
        doc_data (dict): Main document payload stored in `doc-store`.
        selected_box_ids (list): List of IDs representing currently highlighted bounding boxes.
        pdf_viewer_className (str): Current CSS class string for the PDF viewer container..

    Returns:
        tuple: A 4-element tuple updating Plotly figure, JSON tree view component, graph style dict,
            and viewer container class name.
    """
    triggered_inputs = [t["prop_id"] for t in ctx.triggered]
    #debug_print(f"\n[DEBUG] update_display triggered by: {triggered_inputs}")

    fig = go.Figure()
    selected_box_ids = selected_box_ids or []

    if not doc_data or "image_str" not in doc_data:
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=True,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        if " close" not in pdf_viewer_className:
            pdf_viewer_className += " close"

        #debug_print(f"\n[DEBUG] Created empty figure.")
        return fig, "", no_update, pdf_viewer_className

    img_w = doc_data["width"]
    img_h = doc_data["height"]
    scale_x = doc_data["scale_x"]
    scale_y = doc_data["scale_y"]

    fig.add_layout_image(
        dict(
            source=doc_data["image_str"],
            xref="x",
            yref="y",
            x=0,
            y=img_h,
            sizex=img_w,
            sizey=img_h,
            sizing="stretch",
            opacity=1.0,
            layer="below",
            xanchor="left",
            yanchor="bottom",
        )
    )

    if not doc_data.get("blocks", []):
        fig.update_layout(
            xaxis=dict(range=[0, img_w], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(
                range=[img_h, 0],
                showgrid=False,
                zeroline=False,
                visible=False,
                scaleanchor="x",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=True,
            clickmode="event",
            dragmode="select",
            uirevision=doc_data.get("current_page", 0),
        )

        if " close" in pdf_viewer_className:
            pdf_viewer_className = pdf_viewer_className.replace(" close", "")

        return fig, "", {"aspectRatio": f"{img_w} / {img_h}"}, pdf_viewer_className

    # Resolve active page index
    current_page_idx = doc_data.get("current_page", 0)
    doc_json_by_page = doc_data.get("document_json_by_page", {})
    if isinstance(doc_json_by_page, dict):
        active_page_json = (
            doc_json_by_page.get(current_page_idx)
            or doc_json_by_page.get(str(current_page_idx))
            or {}
        )
    else:
        active_page_json = {}

    block_lookup = {}
    blocks_for_lookup = doc_data.get("blocks", [])
    if blocks_for_lookup:
        for b in blocks_for_lookup:
            dict_path = b.get("dict_path", "")

            if dict_path and isinstance(dict_path, list):
                clean_path = tuple(p for p in dict_path if p != "End")
                block_lookup[clean_path] = b
            else:
                k = b.get("key",None) or b.get("text", None)
                if k:
                    block_lookup[(k,)] = b

    page_blocks = [
        b for b in doc_data.get("blocks", [])
        if not b.get("is_duplicate", False)
        and not b.get("is_derived", False)
    ]

    for idx, b in enumerate(page_blocks):
        coords = b.get("coords")
        if not coords or any(c is None for c in coords):
            continue

        block_id = b.get("block_num", "")
        block_type = b.get("type", "text")

        # Resolve clean hovertext representation across raw & postprocessed items
        if "dict_path" in b and b["dict_path"]:
            path_str = " -> ".join(b["dict_path"])
            hovertext = f"Table ({path_str}): {b.get('text', '')}"
        elif block_type == "key_value_pair":
            hovertext = f"{b.get('key', '')}: {b.get('value', '')}"
        elif b.get("key") and b.get("value"):
            hovertext = f"{b['key']}: {b['value']}"
        else:
            hovertext = b.get("text") or b.get("value", "")

        is_selected = block_id in selected_box_ids

        x0, y0 = coords[0] * scale_x, coords[1] * scale_y
        x1, y1 = coords[2] * scale_x, coords[3] * scale_y

        # Style box based on selection state vs. block category
        if is_selected:
            fill_color = "rgba(0, 225, 255, 0.45)"
            line_color = "rgba(0, 150, 255, 1.0)"
            line_width = 3.5
        elif str(block_type) == "table_text_span":
            fill_color = "rgba(255, 255, 0, 0.15)"
            line_color = "rgba(200, 160, 0, 1.0)"
            line_width = 2.5
        elif str(block_type) == "table_text":
            fill_color = "rgba(255, 0, 0, 0.15)"
            line_color = "rgba(200, 0, 0, 1.0)"
            line_width = 2.5
        elif "picture" in str(block_type) or "figure" in str(block_type):
            fill_color = "rgba(255, 0, 255, 0.15)"
            line_color = "rgba(180, 0, 180, 1.0)"
            line_width = 2.5
        elif "header" in str(block_type) or "title" in str(block_type):
            fill_color = "rgba(0, 167, 255, 0.15)"
            line_color = "rgba(0, 100, 200, 1.0)"
            line_width = 2.5
        elif "list" in str(block_type):
            fill_color = "rgba(0, 255, 0, 0.15)"
            line_color = "rgba(0, 160, 0, 1.0)"
            line_width = 2.5
        elif "key_value_pair" in str(block_type):
            fill_color = "rgba(230, 210, 250, 0.15)"
            line_color = "rgba(100, 50, 150, 1.0)"
            line_width = 2.5
        else:
            fill_color = "rgba(167, 167, 0, 0.15)"
            line_color = "rgba(110, 130, 0, 1.0)"
            line_width = 2.5

        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                fill="toself",
                fillcolor=fill_color,
                mode="lines",
                line=dict(color=line_color, width=line_width),
                customdata=[block_id] * 5,
                meta=str(block_type),
                hovertext=hovertext,
                hovertemplate="%{hovertext}<extra></extra>",
                name="",
                showlegend=False,
                selected=dict(marker=dict(opacity=1)),
                unselected=dict(marker=dict(opacity=1)),
            )
        )

    fig.update_layout(
        xaxis=dict(range=[0, img_w], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(
            range=[img_h, 0],
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor="x",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=True,
        clickmode="event",
        dragmode="select",
        uirevision=doc_data.get("current_page", 0),
    )

    if " close" in pdf_viewer_className:
        pdf_viewer_className = pdf_viewer_className.replace(" close", "")

    clean_json = sanitize_json_for_display(active_page_json)
    json_display_str = render_json_tree(clean_json, blocks_lookup=block_lookup)

    return fig, json_display_str, {"aspectRatio": f"{img_w} / {img_h}"}, pdf_viewer_className


# Save Trigger & Overwrite Warning Queue Manager
@app.callback(
    Output("gui-overwriteModalWindow", "className"),
    Output("gui-backdrop", "className", allow_duplicate=True),
    Output("overwrite-modal-message", "children"),
    Output("overwrite-queue-store", "data"),
    Input("btn-save", "n_clicks"),
    Input("btn-save-metadata", "n_clicks"),
    Input("btn-confirm-overwrite-modal", "n_clicks"),
    Input("btn-cancel-overwrite-modal", "n_clicks"),
    Input("btn-close-overwrite-modal", "n_clicks"),
    State("json-output-path", "value"),
    State("export-pdf-check", "value"),
    State("pdf-output-path", "value"),
    State("overwrite-queue-store", "data"),
    prevent_initial_call=True,
)
def manage_save_and_overwrites(
    save_clicks, save_md_clicks, confirm_clicks, cancel_clicks, close_clicks,
    json_path, export_pdf_check, pdf_output_path, queue_data
):
    """Manages file save triggers, path conflict checks, and the overwrite confirmation queue.

    Args:
        save_clicks (int): Click count for the save button.
        save_md_clicks (int): Click count for the save JSON and metadata button.
        confirm_clicks (int): Click count for the confirm overwrite button.
        cancel_clicks (int): Click count for the cancel modal button.
        close_clicks (int): Click count for the modal exit button.
        json_path (str): Currently specified JSON output file path.
        export_pdf_check (list): Checked value array for PDF export checkbox.
        pdf_output_path (str): Currently specified PDF output file path.
        queue_data (dict): Current overwrite queue stored in `overwrite-queue-store`.

    Returns:
        tuple: Updates overwrite modal class name, backdrop class name, modal message, and store payload.
    """
    triggered = ctx.triggered_id
    queue_data = queue_data or {"queue": [], "confirmed": []}

    # User clicks "Save" or "Save Metadata" -> inspect active output paths for collisions
    if triggered in ["btn-save", "btn-save-metadata"]:
        pending_queue = []
        confirmed_files = []

        # Check JSON output path
        if json_path and json_path.strip():
            j_path = os.path.abspath(json_path.strip())
            if os.path.exists(j_path):
                pending_queue.append({"type": "JSON", "path": j_path})
            else:
                confirmed_files.append({"type": "JSON", "path": j_path})

            # If triggered by save metadata button, also queue/confirm the metadata file path
            if triggered == "btn-save-metadata":
                base, ext = os.path.splitext(j_path)
                md_path = f"{base}_metadata{ext}"
                if os.path.exists(md_path):
                    pending_queue.append({"type": "Metadata", "path": md_path})
                else:
                    confirmed_files.append({"type": "Metadata", "path": md_path})

        # Check PDF output path if export checkbox is active
        should_export_pdf = "export_pdf" in export_pdf_check if export_pdf_check else False
        if should_export_pdf and pdf_output_path and pdf_output_path.strip():
            p_path = os.path.abspath(pdf_output_path.strip())
            if os.path.exists(p_path):
                pending_queue.append({"type": "PDF", "path": p_path})
            else:
                confirmed_files.append({"type": "PDF", "path": p_path})

        # If files exist, open modal for the first item in queue
        if pending_queue:
            item = pending_queue[0]
            msg = f"The {item['type']} file already exists at:\n{item['path']}\n\nDo you want to overwrite it?"
            new_store = {"queue": pending_queue, "confirmed": confirmed_files}
            return "gui__overwriteModalWindow open", "gui__backdrop open", msg, new_store

        # No conflicts -> signal immediate write execution
        new_store = {"queue": [], "confirmed": confirmed_files, "execute_save": True}
        return "gui__overwriteModalWindow close", "gui__backdrop close", "", new_store

    # User clicks "Overwrite" in modal
    if triggered == "btn-confirm-overwrite-modal":
        current_queue = queue_data.get("queue", [])
        confirmed_list = queue_data.get("confirmed", [])

        if current_queue:
            approved_item = current_queue.pop(0)
            confirmed_list.append(approved_item)

        # If more conflicting files remain, prompt for the next one
        if current_queue:
            next_item = current_queue[0]
            msg = f"The {next_item['type']} file already exists at:\n{next_item['path']}\n\nDo you want to overwrite it?"
            new_store = {"queue": current_queue, "confirmed": confirmed_list}
            return "gui__overwriteModalWindow open", "gui__backdrop open", msg, new_store

        # All collisions handled -> signal save execution
        new_store = {"queue": [], "confirmed": confirmed_list, "execute_save": True}
        return "gui__overwriteModalWindow close", "gui__backdrop close", "", new_store

    # User cancels/closes modal -> execute save only for non-conflicting/previously confirmed files
    if triggered in ["btn-cancel-overwrite-modal", "btn-close-overwrite-modal"]:
        current_queue = queue_data.get("queue", [])
        confirmed_list = queue_data.get("confirmed", [])

        # Drop remaining unconfirmed items from queue and proceed with safe writes
        new_store = {"queue": [], "confirmed": confirmed_list, "execute_save": True}
        return "gui__overwriteModalWindow close", "gui__backdrop close", "", new_store

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update


# Executable File Writer
@app.callback(
    Output("btn-save", "children"),
    Output("btn-save-metadata", "children"),
    Input("overwrite-queue-store", "data"),
    State("doc-store", "data"),
    prevent_initial_call=True,
)
def execute_verified_save(queue_data, doc_data):
    """Executes file write operations to write confirmed JSON, Metadata, and PDF files to disk.

    Args:
        queue_data (dict): Overwrite queue payload containing confirmed save target paths.
        doc_data (dict): Main document payload stored in `doc-store`.

    Returns:
        tuple: Updated text labels for the save action buttons.
    """
    if not queue_data or not queue_data.get("execute_save"):
        return dash.no_update, dash.no_update

    confirmed_items = queue_data.get("confirmed", [])
    saved_json = False
    saved_meta = False

    # Check queue contents to see which action was requested/confirmed
    has_json_or_pdf = any(item.get("type") in ["JSON", "PDF"] for item in confirmed_items)
    has_metadata = any(item.get("type") == "Metadata" for item in confirmed_items)

    current_page_idx = doc_data.get("current_page", 0) if doc_data else 0

    # Prepare JSON string
    doc_json_by_page = doc_data.get("document_json_by_page", {}) if doc_data else {}
    if isinstance(doc_json_by_page, dict):
        active_page_json = (
            doc_json_by_page.get(current_page_idx)
            or doc_json_by_page.get(str(current_page_idx))
            or {}
        )
    else:
        active_page_json = {}

    clean_json = sanitize_json_for_display(active_page_json)
    json_str = json.dumps(clean_json, indent=2, ensure_ascii=False)

    # Prepare Metadata string
    doc_metadata_by_page = doc_data.get("document_metadata_by_page", {}) if doc_data else {}
    if isinstance(doc_metadata_by_page, dict):
        active_page_metadata = (
            doc_metadata_by_page.get(current_page_idx)
            or doc_metadata_by_page.get(str(current_page_idx))
            or {}
        )
    else:
        active_page_metadata = {}

    clean_metadata = sanitize_json_for_display(active_page_metadata)
    metadata_str = json.dumps(clean_metadata, indent=2, ensure_ascii=False)

    for item in confirmed_items:
        file_type = item.get("type")
        target_path = item.get("path")

        # Write JSON file
        if file_type == "JSON" and json_str:
            try:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(json_str)
                saved_json = True
            except Exception as e:
                print(f"Error saving JSON file: {e}")

        # Write Metadata file
        elif file_type == "Metadata" and metadata_str:
            try:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(metadata_str)
                saved_meta = True
            except Exception as e:
                print(f"Error saving Metadata file: {e}")

        # Write PDF file using cached rendered images
        elif file_type == "PDF" and doc_data:
            try:
                out_pdf = fitz.open()
                pages_dict = doc_data.get("pages", {})

                if pages_dict:
                    sorted_keys = sorted(pages_dict.keys(), key=lambda k: int(k))
                    for p_key in sorted_keys:
                        img_str = pages_dict[p_key].get("image_str", "")
                        if img_str and img_str.startswith("data:image"):
                            base64_data = img_str.split(",")[1]
                            img_bytes = base64.b64decode(base64_data)
                            img_pil = Image.open(io.BytesIO(img_bytes))
                            w, h = img_pil.size

                            page = out_pdf.new_page(width=w, height=h)
                            page.insert_image(page.rect, stream=img_bytes)

                elif "image_str" in doc_data:
                    img_str = doc_data["image_str"]
                    if img_str and img_str.startswith("data:image"):
                        base64_data = img_str.split(",")[1]
                        img_bytes = base64.b64decode(base64_data)
                        img_pil = Image.open(io.BytesIO(img_bytes))
                        w, h = img_pil.size

                        page = out_pdf.new_page(width=w, height=h)
                        page.insert_image(page.rect, stream=img_bytes)

                if len(out_pdf) > 0:
                    out_pdf.save(target_path, garbage=4, deflate=True)
                    saved_json = True
                out_pdf.close()

            except Exception as e:
                print(f"Error saving PDF file: {e}")

    save_label = "Saved!" if (saved_json and has_json_or_pdf and not has_metadata) else no_update
    metadata_label = "Saved!" if (saved_meta and has_metadata) else no_update
    return save_label, metadata_label


# Consolidated Form Field & Button Toggle Callback
@app.callback(
    Output("pdf-output-path", "disabled"),
    Output("btn-browse-pdf-output", "disabled"),
    Output("btn-run-docling", "disabled"),
    Output("btn-export-raw", "disabled"),
    Input("export-pdf-check", "value"),
    Input("doc-store", "data"),
    Input("json-output-path", "value"),
)
def manage_ui_control_states(export_pdf, doc_data, json_path):
    """Enables or disables action buttons and input fields based on form validation state.

    Args:
        export_pdf (list): Selected values in the PDF export checkbox list.
        doc_data (dict): Document payload stored in `doc-store`.
        json_path (str): Currently typed JSON output path string.

    Returns:
        tuple: Boolean disabled flags for PDF output path, browse PDF button, run extractor button, and raw export button.
    """
    pdf_out_disabled = not bool(export_pdf)

    has_pdf = bool(doc_data and "image_str" in doc_data)
    has_json_path = bool(json_path and json_path.strip())
    action_disabled = not (has_pdf and has_json_path)

    return pdf_out_disabled, pdf_out_disabled, action_disabled, action_disabled


# ---- DASH FILE EXPLORER MODAL CALLBACKS ----

# File Explorer Modal Launcher and File Action Handler
@app.callback(
    Output('pdf-input-path', 'value', allow_duplicate=True),
    Output('json-output-path', 'value', allow_duplicate=True),
    Output('pdf-output-path', 'value', allow_duplicate=True),
    Output('explorer-config-store', 'data'),
    Output('gui-fileExplorerWindow', 'className'),
    Output('gui-backdrop', 'className', allow_duplicate=True),
    Output('gui-fileExplorer-filename-bar-error-text', 'children'),
    Output('gui-fileExplorer-filename-bar-error', 'className'),
    Output('explorer-state-store', 'data', allow_duplicate=True),
    Output('gui-fileExplorer-filename-bar-dropdown-widget', 'value', allow_duplicate=True),

    Input('btn-browse-pdf-input', 'n_clicks'),
    Input('btn-browse-json-output', 'n_clicks'),
    Input('btn-browse-pdf-output', 'n_clicks'),
    Input('fileExplorer_actionButton', 'n_clicks'),
    Input('fileExplorer_cancelButton', 'n_clicks'),
    Input('gui-fileExplorer-header-exit-button', 'n_clicks'),
    Input('file-explorer-double-click-trigger', 'data'),

    State('explorer-state-store', 'data'),
    State('explorer-config-store', 'data'),
    State('gui-fileExplorer-filename', 'value'),
    prevent_initial_call=True
)
def manage_modal_and_submit(
    b1, b2, b3, action_clicks, cancel_clicks, exit_clicks,
    double_click_trigger, explorer_state, config_state, typed_filename,
):
    """Manages opening, closing, file selection, and submission for the custom file explorer modal.

    Args:
        b1 (int): Click count for browse input PDF button.
        b2 (int): Click count for browse output JSON button.
        b3 (int): Click count for browse output PDF button.
        action_clicks (int): Click count for file explorer primary action button.
        cancel_clicks (int): Click count for file explorer cancel button.
        exit_clicks (int): Click count for file explorer window close button.
        double_click_trigger (int): Trigger counter fired on item double-clicks.
        explorer_state (dict): Active directory state payload stored in `explorer-state-store`.
        config_state (dict): Explorer configuration payload stored in `explorer-config-store`.
        typed_filename (str): Current string in the file explorer filename input field.

    Returns:
        tuple: A 10-element tuple updating input paths, explorer configs, window classes, errors, and state.
    """
    triggered = ctx.triggered_id
    if not triggered:
        return (no_update,) * 10

    # Open Modal from Browse buttons
    if triggered in ['btn-browse-pdf-input', 'btn-browse-json-output', 'btn-browse-pdf-output']:
        configs = {
            'btn-browse-pdf-input': {
                'target': 'pdf_input',
                'ext': 'pdf',
                'title': 'Select Input PDF',
                'action_text': 'Open'
            },
            'btn-browse-json-output': {
                'target': 'json_output',
                'ext': 'json',
                'title': 'Provide Filename for JSON Output',
                'action_text': 'Save'
            },
            'btn-browse-pdf-output': {
                'target': 'pdf_output',
                'ext': 'pdf',
                'title': 'Provide Filename for PDF Output',
                'action_text': 'Save'
            },
        }
        config = configs[triggered]
        target_ext = config['ext'] # 'pdf' or 'json'

        # Clean double click tracking state when opening a new dialog
        clean_state = explorer_state or {'cwd': os.getcwd(), 'filename': '', 'history': [os.getcwd()]}
        clean_state['last_clicked'] = None
        clean_state['last_click_time'] = 0

        # If there is an existing filename, make sure the extension matches the current browser context
        current_filename = clean_state.get('filename', '')
        if current_filename and target_ext:
            base_name, _ = os.path.splitext(current_filename)
            clean_state['filename'] = f"{base_name}.{target_ext}"

        return (
            no_update, no_update, no_update,
            config,
            'gui__fileExplorerWindow open',
            'gui__backdrop open',
            '',
            'genericInput__error filenameBar__error close',
            clean_state,
            target_ext,
        )

    # Close Modal on Cancel or Exit
    if triggered in ['fileExplorer_cancelButton', 'gui-fileExplorer-header-exit-button']:
        return (
            no_update, no_update, no_update,
            no_update,
            'gui__fileExplorerWindow close',
            'gui__backdrop close',
            '',
            'genericInput__error filenameBar__error close',
            no_update,
            no_update,
        )

    # Confirm / Select Action
    if triggered in ['fileExplorer_actionButton', 'file-explorer-double-click-trigger']:
        explorer_state = explorer_state or {}
        cwd = Path(explorer_state.get('cwd', os.getcwd()))
        target = config_state.get('target') if config_state else ''
        expected_ext = config_state.get('ext') if config_state else None

        # On double-click, explicitly grab filename from explorer_state first
        if triggered == 'file-explorer-double-click-trigger':
            filename = explorer_state.get('filename', '').strip()
        else:
            filename = (typed_filename or explorer_state.get('filename', '')).strip()

        if target in ['json_output', 'pdf_output']:
            if not filename:
                return (
                    no_update, no_update, no_update,
                    no_update,
                    'gui__fileExplorerWindow open',
                    'gui__backdrop open',
                    'Please specify a filename.',
                    'genericInput__error filenameBar__error open',
                    no_update,
                    no_update,
                )

            if expected_ext:
                req_ext = f".{expected_ext.lower().lstrip('.')}"
                base_name, current_ext = os.path.splitext(filename)
                if current_ext.lower() != req_ext:
                    filename = f"{base_name}{req_ext}"

            full_path = cwd / filename

            can_create, error_msg = check_file_creatable(full_path)
            if not can_create:
                return (
                    no_update, no_update, no_update,
                    no_update,
                    'gui__fileExplorerWindow open',
                    'gui__backdrop open',
                    error_msg,
                    'genericInput__error filenameBar__error open',
                    no_update,
                    no_update,
                )

            json_out = str(full_path) if target == 'json_output' else no_update
            pdf_out = str(full_path) if target == 'pdf_output' else no_update

            return (
                no_update, json_out, pdf_out,
                no_update,
                'gui__fileExplorerWindow close',
                'gui__backdrop close',
                '',
                'genericInput__error filenameBar__error close',
                no_update,
                no_update,
            )

        elif target == 'pdf_input':
            filename = filename or (explorer_state.get('filename', '') if explorer_state else '')
            full_path = str(cwd / filename) if filename else str(cwd)
            return (
                full_path, no_update, no_update,
                no_update,
                'gui__fileExplorerWindow close',
                'gui__backdrop close',
                '',
                'genericInput__error filenameBar__error close',
                no_update,
                no_update,
            )

    return (no_update,) * 10


# File Explorer State Handler
@app.callback(
    Output('explorer-state-store', 'data', allow_duplicate=True),
    Output('file-explorer-double-click-trigger', 'data'),
    Input('gui-fileExplorer-directory-bar-backButton', 'n_clicks'),
    Input('gui-fileExplorer-directory-bar-upButton', 'n_clicks'),
    Input('gui-fileExplorer-directory-bar-cwd-widget', 'n_submit'),
    Input({'type': 'fileList', 'index': ALL}, 'n_clicks'),
    State('explorer-config-store', 'data'),
    State('gui-fileExplorer-directory-bar-cwd-widget', 'value'),
    State('explorer-state-store', 'data'),
    State('file-explorer-double-click-trigger', 'data'),
    prevent_initial_call=True
)
def update_explorer_state(
    back_clicks, up_clicks, cwd_input, item_clicks, config_data,
    cwd_input_val, current_state, trigger_count,
):
    """Updates file explorer navigation history, current directory, and double-click triggers.

    Args:
        back_clicks (int): Click count for directory back button.
        up_clicks (int): Click count for parent directory button.
        cwd_input (int): Submit counter for manual working directory input field.
        item_clicks (list): Click counters for items in the workspace file list.
        config_data (dict): Config stored in `explorer-config-store`.
        cwd_input_val (str): Current string in directory path input field.
        current_state (dict): Active state dictionary stored in `explorer-state-store`.
        trigger_count (int): Counter tracking double-click trigger events.

    Returns:
        tuple: Updated state dictionary for `explorer-state-store` and double-click trigger counter.
    """
    triggered = ctx.triggered_id
    if not triggered:
        return no_update, no_update

    trigger_count = trigger_count or 0

    state = current_state or {
        'cwd': os.getcwd(),
        'filename': '',
        'history': [os.getcwd()],
        'last_clicked': None,
        'last_click_time': 0,
    }

    current_cwd = Path(state['cwd'])

    if triggered == 'explorer-config-store':
        new_state = {
            'cwd': state['cwd'],
            'filename': state['filename'],
            'history': update_folder_history(state.get('history', []), state['cwd']),
            'last_clicked': None,
            'last_click_time': 0,
        }
        return new_state, no_update

    if triggered == 'gui-fileExplorer-directory-bar-backButton':
        new_history = update_folder_history(state['history'], state['cwd'], is_back_action=True)
        new_state = {
            'cwd': new_history[-1],
            'filename': '',
            'history': new_history,
            'last_clicked': None,
            'last_click_time': 0,
        }
        return new_state, no_update

    if triggered == 'gui-fileExplorer-directory-bar-upButton':
        parent_dir = str(current_cwd.parent)
        new_state = {
            'cwd': parent_dir,
            'filename': '',
            'history': update_folder_history(state['history'], parent_dir),
            'last_clicked': None,
            'last_click_time': 0,
        }
        return new_state, no_update

    if triggered == 'gui-fileExplorer-directory-bar-cwd-widget':
        target_path = Path(cwd_input_val)
        if target_path.is_dir():
            new_state = {
                'cwd': str(target_path),
                'filename': '',
                'history': update_folder_history(state['history'], str(target_path)),
                'last_clicked': None,
                'last_click_time': 0,
            }
            return new_state, no_update
        return no_update

    if isinstance(triggered, dict) and triggered.get('type') == 'fileList':
        click_val = ctx.triggered[0].get('value')
        if not click_val:
            return no_update, no_update

        selected_item = triggered['index']
        target_path = current_cwd / selected_item

        if target_path.is_dir():
            new_state = {
                'cwd': str(target_path),
                'filename': '',
                'history': update_folder_history(state['history'], str(target_path)),
                'last_clicked': None,
                'last_click_time': 0,
            }
            return new_state, no_update
        elif target_path.is_file():
            now = time.time()
            is_double = (
                state.get('last_clicked') == selected_item and
                (now - state.get('last_click_time', 0)) < 0.4
            )
            new_state = dict(state)
            state['filename'] = selected_item
            state['last_clicked'] = selected_item
            state['last_click_time'] = now

            if is_double:
                return state, trigger_count + 1

            return state, no_update

    return state, no_update


# File Explorer UI Renderer
@app.callback(
    Output('gui-fileExplorer-workspace-file-list', 'children'),
    Output('gui-fileExplorer-directory-bar-cwd-widget', 'value'),
    Output('gui-fileExplorer-filename', 'value'),
    Output('gui-fileExplorer-header-title-text', 'children'),
    Output('gui-fileExplorer-actionButton-text', 'children'),
    Output('gui-fileExplorer-directory-bar-backButton', 'disable_n_clicks'),
    Output('gui-fileExplorer-directory-bar-upButton', 'disable_n_clicks'),
    Output('gui-fileExplorer-directory-bar-error-text', 'children'),
    Output('gui-fileExplorer-directory-bar-error', 'className'),
    Input('explorer-state-store', 'data'),
    Input('gui-fileExplorer-filename-bar-dropdown-widget', 'value'),
    State('explorer-config-store', 'data')
)
def render_explorer_ui(explorer_state, dropdown_ext, config_data):
    """Renders workspace components, navigation buttons, and error banners for file explorer.

    Args:
        explorer_state (dict): State payload stored in `explorer-state-store`.
        dropdown_ext (str): Selected extension filter option from the dropdown widget.
        config_data (dict): Configuration stored in `explorer-config-store`.

    Returns:
        tuple: A 9-element tuple updating file list HTML elements, directory bar, filename, header title,
            action button text, back/up button states, and error text banner components.
    """
    if not explorer_state:
        return (no_update,) * 9

    cwd_path = Path(explorer_state['cwd'])
    config = config_data or {}

    default_ext = config.get('ext', 'all')
    ext_filter = dropdown_ext if dropdown_ext and dropdown_ext != 'all' else default_ext
    file_elements, error_msg = get_directory_items(cwd_path, ext_filter=ext_filter)

    history = explorer_state.get('history', [])
    back_disabled = len(history) <= 1
    up_disabled = cwd_path.parent == cwd_path

    dir_error_class = "gui__error open" if error_msg else "gui__error close"

    return (
        file_elements,
        str(cwd_path),
        explorer_state.get('filename', ''),
        config.get('title', 'Browse Files...'),
        config.get('action_text', 'Open'),
        back_disabled,
        up_disabled,
        error_msg,
        dir_error_class
    )


# Handles opening and closing of schema modal
@app.callback(
    Output("gui-schemaModalWindow", "className"),
    Output("gui-backdrop", "className", allow_duplicate=True),
    Output("schema-text-modal", "value"),
    Output("schema-text", "data"),
    Input("btn-open-schema-modal", "n_clicks"),
    Input("btn-close-schema-modal", "n_clicks"),
    Input("btn-cancel-schema-modal", "n_clicks"),
    Input("btn-apply-schema-modal", "n_clicks"),
    State("schema-text-modal", "value"),
    State("schema-text", "data"),
    prevent_initial_call=True
)
def handle_schema_modal(open_clicks, close_clicks, cancel_clicks, apply_clicks, modal_val, stored_val):
    """Manages opening, closing, and applying extraction schema definitions via modal dialog.

    Args:
        open_clicks (int): Click count for open schema modal button.
        close_clicks (int): Click count for modal close exit button.
        cancel_clicks (int): Click count for modal cancel button.
        apply_clicks (int): Click count for modal apply schema button.
        modal_val (str): Text content from the modal textarea input.
        stored_val (str): Schema string stored in `schema-text` store.

    Returns:
        tuple: CSS class for schema modal, backdrop CSS class, modal text area value, and stored schema data.
    """
    triggered = ctx.triggered_id

    if triggered == "btn-open-schema-modal":
        return "gui__schemaModalWindow open", "gui__backdrop open", stored_val or "", dash.no_update

    if triggered in ["btn-close-schema-modal", "btn-cancel-schema-modal"]:
        return "gui__schemaModalWindow close", "gui__backdrop close", dash.no_update, dash.no_update

    if triggered == "btn-apply-schema-modal":
        return "gui__schemaModalWindow close", "gui__backdrop close", dash.no_update, modal_val

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update


# Resets selected boxes on page change
@app.callback(
    Output("selected-boxes-store", "data", allow_duplicate=True),
    Input("doc-store", "data"),
    State("selected-boxes-store", "data"),
    prevent_initial_call=True,
)
def reset_selection_on_page_change(doc_data, selected_blocks):
    """Resets the bounding box selections store whenever the active document or page changes.

    Args:
        doc_data (dict): Document payload stored in `doc-store`.

    Returns:
        list: Empty list resetting `selected-boxes-store`.
    """
    if selected_blocks:
        return []
    return no_update


@app.callback(
    Output("doc-store", "data", allow_duplicate=True),
    Input({'type': 'json-edit-key', 'path': ALL}, 'value'),
    Input({'type': 'json-edit-value', 'path': ALL}, 'value'),
    State("doc-store", "data"),
    prevent_initial_call=True,
)
def update_json_from_tree_input(all_keys, all_vals, doc_data):
    """Updates document store JSON data structures and block dict_paths in response to tree view edits.

    Args:
        all_keys (list): Values array from all pattern-matched JSON tree key input fields.
        all_vals (list): Values array from all pattern-matched JSON tree value input fields.
        doc_data (dict): Document payload stored in `doc-store`.

    Returns:
        dict: Updated document state payload dictionary.
    """
    triggered_ctx = ctx.triggered
    if not triggered_ctx or not doc_data:
        return no_update

    triggered_id = ctx.triggered_id
    if not isinstance(triggered_id, dict) or 'path' not in triggered_id:
        return no_update

    try:
        path = json.loads(triggered_id['path'])
    except Exception as e:
        print(f"Error parsing JSON tree path: {e}")
        return no_update

    new_val = ctx.triggered[0]['value']
    is_key_edit = (triggered_id.get('type') == 'json-edit-key')

    current_page_idx = doc_data.get("current_page", 0)
    page_key_str = str(current_page_idx)

    doc_json_by_page = doc_data.get("document_json_by_page", {})
    if not isinstance(doc_json_by_page, dict):
        return no_update

    active_page_json = doc_json_by_page.get(current_page_idx) or doc_json_by_page.get(page_key_str, {})
    if not active_page_json:
        return no_update

    curr = active_page_json
    for key in path [:-1]:
        if isinstance(curr, dict) and key in curr:
            curr = curr[key]
        elif isinstance(curr, list) and isinstance(key, int) and 0 <= key < len(curr):
            curr = curr[key]
        else:
            break

    last_key = path[-1]
    pages_dict = doc_data.get("pages", {})
    page_info = pages_dict.get(page_key_str, {})
    blocks = page_info.get("blocks", []) or doc_data.get("blocks", [])

    if is_key_edit:
        old_key = last_key
        new_key = new_val.strip()
        if not new_key or old_key == new_key:
            return no_update

        # Update dict_path in underlying blocks where this key appears
        for b in blocks:
            d_path = b.get("dict_path")
            if d_path and isinstance(d_path, list):
                updated_path = [new_key if p == old_key else p for p in d_path]
                b["dict_path"] = updated_path

        new_page_json = build_page_json_from_blocks(blocks)
        doc_json_by_page[page_key_str] = new_page_json
        doc_data["document_json_by_page"] = doc_json_by_page
        if "pages" in doc_data and page_key_str in doc_data["pages"]:
            doc_data["pages"][page_key_str]["blocks"] = blocks
        else:
            doc_data["blocks"] = blocks

    else:
        existing_val = None
        if isinstance(curr, dict):
            existing_val = curr.get(last_key)
        elif isinstance(curr, list) and isinstance(last_key, int) and 0 <= last_key < len(curr):
            existing_val = curr[last_key]

        if str(existing_val) == str(new_val):
            return no_update

        if isinstance(curr, dict) and last_key in curr:
            curr[last_key] = new_val
        elif isinstance(curr, list) and isinstance(last_key, int) and 0 <= last_key < len(curr):
            curr[last_key] = new_val
        else:
            return no_update

        doc_json_by_page[page_key_str] = active_page_json
        doc_data["document_json_by_page"] = doc_json_by_page

    return doc_data


@app.callback(
    Output("pdf-layer-dropdown-menu", "className"),
    Input("btn-toggle-layer-dropdown", "n_clicks"),
    State("pdf-layer-dropdown-menu", "className"),
    prevent_initial_call=True,
)
def toggle_layer_dropdown(n_clicks, current_class):
    """Toggles visibility class name for the PDF layer filter dropdown menu.

    Args:
        n_clicks (int): Click count for layer filter toggle button.
        current_class (str): Current CSS class name string for layer menu component.

    Returns:
        str: Updated CSS class string toggling menu open or closed.
    """
    if "close" in (current_class or ""):
        return "pdf-layer-dropdown-menu open"
    return "pdf-layer-dropdown-menu close"


@app.callback(
    Output({'type': 'json-lazy-content', 'path': MATCH}, 'children'),
    Input({'type': 'json-lazy-summary', 'path': MATCH}, 'n_clicks'),
    State('doc-store', 'data'),
    prevent_initial_call=True
)
def render_incremental_lazy_json_branch(n_clicks, doc_data):
    debug_print(f"\n--- LAZY CALLBACK TRIGGERED ---")
    debug_print(f"is_open: {n_clicks}")
    debug_print(f"triggered_id: {ctx.triggered_id}")

    if not n_clicks:
        debug_print("-> Ignored: No clicks registered.")
        return dash.no_update

    triggered_id = ctx.triggered_id
    if not triggered_id or not triggered_id.get('path'):
        debug_print("-> Error: No valid triggered path found.")
        return dash.no_update

    path_str = triggered_id.get('path')
    try:
        path = json.loads(path_str)
        debug_print(f"Parsed Path: {path}")
    except Exception as e:
        debug_print(f"-> Error parsing path json: {e}")
        return dash.no_update

    # Retrieve active page JSON from doc-store
    current_page_idx = doc_data.get("current_page", 0) if doc_data else 0
    doc_json_by_page = doc_data.get("document_json_by_page", {}) if doc_data else {}
    if isinstance(doc_json_by_page, dict):
        active_page_json = (
            doc_json_by_page.get(current_page_idx)
            or doc_json_by_page.get(str(current_page_idx))
            or {}
        )
    else:
        active_page_json = {}

    # Traverse document JSON to find sub-data at this path
    curr = active_page_json
    for key in path:
        if isinstance(curr, dict) and key in curr:
            curr = curr[key]
        elif isinstance(curr, list) and isinstance(key, int) and 0 <= key < len(curr):
            curr = curr[key]
        else:
            debug_print(f"-> Error: Path traversal failed at key '{key}' in structure of type {type(curr)}")
            curr = None
            break

    if curr is None:
        debug_print("-> Error: Resolved target data is None.")
        return html.Div("Error: Data node not found", style={"color": "red"})

    debug_print(f"Successfully resolved data type: {type(curr)}, item count: {len(curr) if isinstance(curr, (dict, list)) else 'N/A'}")

    # Rebuild blocks lookup for styling references
    block_lookup = {}
    blocks_for_lookup = doc_data.get("blocks", []) if doc_data else []
    for b in blocks_for_lookup:
        dict_path = b.get("dict_path", "")
        if dict_path and isinstance(dict_path, list):
            clean_path = tuple(p for p in dict_path if p != "End")
            block_lookup[clean_path] = b

    try:
        rendered_subtree = render_json_tree(
            curr,
            path=path,
            blocks_lookup=block_lookup,
            max_depth=len(path) + 1,
            current_depth=len(path)
        )
        debug_print("-> Subtree rendered successfully.")
        return rendered_subtree
    except Exception as e:
        debug_print(f"-> Exception during render_json_tree: {e}")
        import traceback
        traceback.debug_print_exc()
        return html.Div(f"Error rendering tree: {str(e)}", style={"color": "red"})


@app.callback(
    Output("selected-boxes-store", "data", allow_duplicate=True),
    Input("blank-click-trigger", "n_clicks"),
    prevent_initial_call=True,
)
def deselect_all_on_blank_click(n_clicks):
    """Clears all selected bounding boxes when clicking blank graph space.

    Args:
        n_clicks (int): Click count incremented when the hidden trigger button is clicked.

    Returns:
        list: An empty list to reset the selected boxes store.
    """
    if n_clicks:
        return []
    return no_update

# ---- CLIENTSIDE CALLBACKS ----


clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            const workspace = document.getElementById('gui-fileExplorer-workspace');
            if (workspace) {
                workspace.scrollTop = 0;
            }
        }, 0);

        return window.dash_clientside.no_update;
    }
    """,
    Output("gui-fileExplorer-workspace-file-list", "aria-label"),
    Input("gui-fileExplorer-workspace-file-list", "children"),
    prevent_initial_call=True,
)
def reset_file_explorer_scroll(children):
    """Reset file explorer workspace scroll position to the top upon file list updates.

    Args:
        children: Updated file list elements rendered inside the workspace.

    Returns:
        dash.no_update: Maintains existing component properties.
    """


clientside_callback(
    """
    function(filterValues) {
        const graphDiv = document.querySelector('#pdf-graph .js-plotly-plot') || document.getElementById('pdf-graph');
        if (!graphDiv || !graphDiv.data) {
            return window.dash_clientside.no_update;
        }

        const allowed = new Set(filterValues || []);
        const trueIndices = [];
        const falseIndices = [];

        for (let i = 0; i < graphDiv.data.length; i++) {
            const trace = graphDiv.data[i];
            const bType = trace.meta;
            
            // If trace has no meta (e.g., background image), keep it visible
            if (!bType || allowed.has(bType)) {
                trueIndices.push(i);
            } else {
                falseIndices.push(i);
            }
        }

        if (trueIndices.length > 0) {
            Plotly.restyle(graphDiv, {visible: true}, trueIndices);
        }
        if (falseIndices.length > 0) {
            Plotly.restyle(graphDiv, {visible: false}, falseIndices);
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output("pdf-graph", "className", allow_duplicate=True),
    Input("block-type-filter", "value"),
    prevent_initial_call=True,
)
def filter_pdf_layers(filterValues):
    """Dynamically toggle visibility of specific PDF block layers on the Plotly graph.

    Args:
        filterValues (list): List of active block category types selected for display.

    Returns:
        dash.no_update: Maintains existing component properties.
    """


clientside_callback(
    """
    function(n_clicks) {
        if (!window.layerDropdownInitialized) {
            window.layerDropdownInitialized = true;
            document.addEventListener('click', function(event) {
                const wrapper = document.querySelector('.pdf-layer-dropdown-wrapper');
                const menu = document.getElementById('pdf-layer-dropdown-menu');
                
                if (wrapper && menu && !wrapper.contains(event.target)) {
                    if (menu.classList.contains('open')) {
                        const btn = document.getElementById('btn-toggle-layer-dropdown');
                        if (btn) {
                            btn.click();
                        }
                    }
                }
            });
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("pdf-layer-dropdown-menu", "style"),
    Input("btn-toggle-layer-dropdown", "n_clicks"),
    prevent_initial_call=False,
)
def initialize_layer_dropdown_outside_click_listener(n_clicks):
    """Initializes a document-level click listener to close the PDF layer dropdown.

    Detects clicks outside of the layer dropdown wrapper and automatically
    simulates a toggle click to close the menu when clicking elsewhere on the page.

    Args:
        n_clicks (int): Click count incremented when toggling the layer dropdown button.

    Returns:
        dash.no_update: Maintains existing component style properties.
    """


clientside_callback(
    """
    function(clickData, selectedBoxIds) {
        if (!window.modifierKeyTrackerInit) {
            window.modifierKeyTrackerInit = true;
            window.lastCtrlKey = false;
            window.lastShiftKey = false;
            
            document.addEventListener('click', function(e) {
                window.lastCtrlKey = e.ctrlKey || e.metaKey;
                window.lastShiftKey = e.shiftKey;
            }, true);
        }

        if (window.dash_clientside.callback_context.triggered.some(t => t.prop_id === 'pdf-graph.clickData')) {
            // If clickData is null or points array is empty, user clicked blank space -> Deselect all
            if (!clickData || !clickData.points || clickData.points.length === 0) {
                return [];
            }
            
            const point = clickData.points[0];
            let clickedId = point.customdata;
            if (clickedId === undefined || clickedId === null) {
                return [];
            }
            if (Array.isArray(clickedId)) {
                clickedId = clickedId[0];
            }

            let currentSelected = [...(selectedBoxIds || [])];

            if (window.lastCtrlKey) {
                if (!currentSelected.includes(clickedId)) {
                    currentSelected.push(clickedId);
                }
            } else if (window.lastShiftKey) {
                const index = currentSelected.indexOf(clickedId);
                if (index > -1) {
                    currentSelected.splice(index, 1);
                }
            } else {
                currentSelected = [clickedId];
            }

            return currentSelected;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("selected-boxes-store", "data", allow_duplicate=True),
    Input("pdf-graph", "clickData"),
    State("selected-boxes-store", "data"),
    prevent_initial_call=True,
)
def handle_box_click_selection(clickData, selectedBoxIds):
    """Handle individual bounding box click selections with support for single-click isolation, Ctrl+click additions, Shift+click removals, and empty space deselection.

    Args:
        clickData (dict): Plotly event payload containing clicked point data and modifier key states.
        selectedBoxIds (list): Current list of selected bounding box identifiers.

    Returns:
        list: Updated array of selected block IDs.
    """


clientside_callback(
    """
    function(selectedData, selectedBoxIds) {
        if (!window.dragDirectionInitialized) {
            window.dragDirectionInitialized = true;
            window.dragStartX = 0;
            window.isDraggingGraph = false;
            window.isLeftToRight = true;
            window.dragShiftKey = false;
            
            document.addEventListener('mousedown', function(e) {
                const graph = document.getElementById('pdf-graph');
                if (graph && graph.contains(e.target)) {
                    window.dragStartX = e.clientX;
                    window.isDraggingGraph = true;
                    window.dragShiftKey = e.shiftKey;
                }
            }, true);
            
            document.addEventListener('mouseup', function(e) {
                if (window.isDraggingGraph) {
                    const dragEndX = e.clientX;
                    window.isLeftToRight = (dragEndX >= window.dragStartX);
                    window.isDraggingGraph = false;
                }
            }, true);
        }

        if (window.dash_clientside.callback_context.triggered.some(t => t.prop_id === 'pdf-graph.selectedData')) {
            const graphDiv = document.querySelector('#pdf-graph .js-plotly-plot') || document.getElementById('pdf-graph');

            if (!selectedData || !selectedData.range) {
                return window.dash_clientside.no_update;
            }

            let currentSelected = new Set(selectedBoxIds || []);
            
            const rangeX = selectedData.range.x;
            const rangeY = selectedData.range.y;
            const selXMin = Math.min(rangeX[0], rangeX[1]);
            const selXMax = Math.max(rangeX[0], rangeX[1]);
            const selYMin = Math.min(rangeY[0], rangeY[1]);
            const selYMax = Math.max(rangeY[0], rangeY[1]);

            const newlyMatched = [];

            if (graphDiv && graphDiv.data) {
                graphDiv.data.forEach((trace) => {
                    if (!trace.customdata) return; // Skip background image

                    let bXMin = Math.min(...trace.x);
                    let bXMax = Math.max(...trace.x);
                    let bYMin = Math.min(...trace.y);
                    let bYMax = Math.max(...trace.y);

                    let boxId = trace.customdata;
                    if (Array.isArray(boxId)) boxId = boxId[0];

                    if (window.isLeftToRight) {
                        if (bXMin >= selXMin && bXMax <= selXMax && bYMin >= selYMin && bYMax <= selYMax) {
                            newlyMatched.push(boxId);
                        }
                    } else {
                        const isIntersecting = !(bXMax < selXMin || bXMin > selXMax || bYMax < selYMin || bYMin > selYMax);
                        if (isIntersecting) {
                            newlyMatched.push(boxId);
                        }
                    }
                });
            }

            if (window.dragShiftKey) {
                newlyMatched.forEach(id => currentSelected.delete(id));
            } else {
                newlyMatched.forEach(id => currentSelected.add(id));
            }

            // Instantly clear the selection box overlay using a short timeout to let Plotly finish its draw cycle
            if (graphDiv && window.Plotly) {
                setTimeout(() => {
                    Plotly.relayout(graphDiv, { selections: [] });
                }, 10);
            }

            return Array.from(currentSelected);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("selected-boxes-store", "data", allow_duplicate=True),
    Input("pdf-graph", "selectedData"),
    State("selected-boxes-store", "data"),
    prevent_initial_call=True,
)
def handle_marquee_selection(selectedData, selectedBoxIds):
    """Handle click-and-drag marquee selections, differentiating between window selection (Left-to-Right enclosure) and crossing selection (Right-to-Left intersection).

    Args:
        selectedData (dict): Plotly event payload containing selection rectangle coordinates and points.
        selectedBoxIds (list): Current list of selected bounding box identifiers.

    Returns:
        list: Updated array of selected block IDs.
    """
clientside_callback(
    """
    function(clickData) {
        // Whenever Plotly fires clickData, a valid bounding box was clicked!
        if (clickData && clickData.points && clickData.points.length > 0) {
            window.lastClickWasBox = true;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("blank-click-trigger", "style"), # Dummy output
    Input("pdf-graph", "clickData"),
    prevent_initial_call=True,
)
def track_box_clicks(clickData):
    """Tracks valid bounding box clicks to prevent false-positive blank deselection.

    Sets a global flag (`window.lastClickWasBox`) to true when Plotly successfully
    registers a click on a bounding box trace.

    Args:
        clickData (dict): Plotly event data containing points and metadata of the clicked trace.

    Returns:
        dash.no_update: Maintains existing component properties without modification.
    """


clientside_callback(
    """
    function(graphId) {
        if (!window.blankClickListenerInitialized) {
            window.blankClickListenerInitialized = true;
            window.lastClickWasBox = false;
            
            document.addEventListener('click', function(e) {
                const graph = document.getElementById('pdf-graph');
                if (graph && graph.contains(e.target)) {
                    // Give Plotly a microsecond to fire clickData if a box was hit
                    setTimeout(() => {
                        if (window.lastClickWasBox) {
                            // Reset the flag and do nothing (it was a box click)
                            window.lastClickWasBox = false;
                        } else {
                            // It was true blank space! Trigger deselection.
                            const trigger = document.getElementById('blank-click-trigger');
                            if (trigger) {
                                trigger.click();
                            }
                        }
                    }, 50);
                }
            }, true);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("blank-click-trigger", "className"), # Dummy output
    Input("pdf-graph", "id"),
    prevent_initial_call=False,
)
def initialize_blank_space_listener(graph_id):
    """Initializes a global DOM click listener to handle blank-space deselection.

    Attaches a capturing click event listener to the document to detect clicks
    within the PDF graph. Evaluates whether the click was on blank space (using a
    flag set by valid box clicks) and triggers deselection when appropriate.

    Args:
        graph_id (str): The HTML element ID of the PDF graph component.

    Returns:
        dash.no_update: Maintains existing component properties without modification.
    """


clientside_callback(
    """
    function(selectedBoxIds) {
        const graphDiv = document.querySelector('#pdf-graph .js-plotly-plot') || document.getElementById('pdf-graph');
        if (!graphDiv || !graphDiv.data) {
            return window.dash_clientside.no_update;
        }

        const selected = new Set(selectedBoxIds || []);
        const nTraces = graphDiv.data.length;
        
        const updateFillColors = new Array(nTraces);
        const updateLineColors = new Array(nTraces);
        const updateLineWidths = new Array(nTraces);
        const indices = [];

        for (let i = 0; i < nTraces; i++) {
            const trace = graphDiv.data[i];
            if (!trace.customdata) continue; // Skip background image

            indices.push(i);
            const blockId = Array.isArray(trace.customdata) ? trace.customdata[0] : trace.customdata;
            const isSelected = selected.has(blockId);

            if (isSelected) {
                updateFillColors[i] = 'rgba(0, 225, 255, 0.45)';
                updateLineColors[i] = 'rgba(0, 150, 255, 1.0)';
                updateLineWidths[i] = 3.5;
            } else {
                const bType = trace.meta || 'text';
                let fillColor = 'rgba(167, 167, 0, 0.15)';
                let lineColor = 'rgba(110, 130, 0, 1.0)';
                let lineWidth = 2.5;

                if (bType === 'table_text_span') {
                    fillColor = 'rgba(255, 255, 0, 0.15)'; lineColor = 'rgba(200, 160, 0, 1.0)';
                } else if (bType === 'table_text') {
                    fillColor = 'rgba(255, 0, 0, 0.15)'; lineColor = 'rgba(200, 0, 0, 1.0)';
                } else if (bType.includes('picture') || bType.includes('figure')) {
                    fillColor = 'rgba(255, 0, 255, 0.15)'; lineColor = 'rgba(180, 0, 180, 1.0)';
                } else if (bType.includes('header') || bType.includes('title')) {
                    fillColor = 'rgba(0, 167, 255, 0.15)'; lineColor = 'rgba(0, 100, 200, 1.0)';
                } else if (bType.includes('list')) {
                    fillColor = 'rgba(0, 255, 0, 0.15)'; lineColor = 'rgba(0, 160, 0, 1.0)';
                } else if (bType.includes('key_value_pair')) {
                    fillColor = 'rgba(230, 210, 250, 0.15)'; lineColor = 'rgba(100, 50, 150, 1.0)';
                }

                updateFillColors[i] = fillColor;
                updateLineColors[i] = lineColor;
                updateLineWidths[i] = lineWidth;
            }
        }

        if (indices.length > 0) {
            Plotly.restyle(graphDiv, {
                'fillcolor': updateFillColors,
                'line.color': updateLineColors,
                'line.width': updateLineWidths
            }, indices);
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output("pdf-graph", "className", allow_duplicate=True),
    Input("selected-boxes-store", "data"),
    prevent_initial_call=True,
)
def update_graph_box_styles(selectedBoxIds):
    """Instantly restyle Plotly graph bounding box traces in the browser based on current selection state and block category types.

    Args:
        selectedBoxIds (list): Current list of selected bounding box identifiers.

    Returns:
        dash.no_update: Maintains existing component properties.
    """


# ---- Helper Functions ----

def sanitize_json_for_display(data: Any) -> Any:
    """Recursively simplifies internal dictionary payloads for UI display.

    Strips metadata wrappers (e.g., 'value', 'coords', 'confidence') while
    preserving clean nested key-value structures.

    Args:
        data (Any): Dict, list, or scalar value from document state.

    Returns:
        Any: Sanitized structure suitable for JSON tree display.
    """
    if isinstance(data, dict):
        # Unwrap standard metadata dicts holding 'value' along with 'coords' or 'confidence'
        if "value" in data and len(data) <= 3 and any(k in data for k in ("coords", "confidence")):
            return sanitize_json_for_display(data["value"])

        sanitized = {}
        for k, v in data.items():
            sanitized[k] = sanitize_json_for_display(v)
        return sanitized

    elif isinstance(data, list):
        return [sanitize_json_for_display(item) for item in data]

    return data

def build_page_json_from_blocks(blocks: list[dict]) -> dict:
    """Reconstructs a clean page-level JSON structure from block dictionary paths.

    Args:
        blocks (list[dict]): List of block dictionaries containing key-value data and `dict_path`.

    Returns:
        dict: Reconstructed nested dictionary representing document data for the active page.
    """
    page_json = {}

    for idx, b in enumerate(blocks):
        if b.get("is_duplicate", False):
            continue

        dict_path = b.get("dict_path")
        val = b.get("text", "") or b.get("value", "")

        if dict_path and isinstance(dict_path, list):
            curr = page_json
            clean_path = [p for p in dict_path if p != "End"]

            if clean_path:
                # Traverse down to the parent dictionary/container
                for level in clean_path[:-1]:
                    if level not in curr or not isinstance(curr[level], dict):
                        curr[level] = {}
                    curr = curr[level]

                # Assign the terminal leaf value
                leaf_key = clean_path[-1]
                if leaf_key not in curr:
                    curr[leaf_key] = val
        else:
            # Top-level item (dict_path is None)
            k = b.get("key") or f"Block_{b.get('block_num', idx)}"
            if k in page_json:
                instances = sum(1 for existing in page_json if existing.startswith(k))
                k = f"{k}_{instances + 1:02d}"
            page_json[k] = val

    return page_json


def get_block_type_color(block_type: str) -> str:
    """Returns a CSS-compatible RGB color string corresponding to a PDF block category.

    Args:
        block_type (str): Category string of the block (e.g., 'table_text', 'header').

    Returns:
        str: Formatted CSS RGB string matching graph box visual styling.
    """
    b_type = str(block_type)
    if b_type == "table_text_span":
        return "rgb(200, 160, 0)"     # Gold / Yellow-ish
    elif b_type == "table_text":
        return "rgb(200, 0, 0)"       # Red
    elif "picture" in b_type or "figure" in b_type:
        return "rgb(180, 0, 180)"     # Purple / Magenta
    elif "header" in b_type or "title" in b_type:
        return "rgb(0, 100, 200)"     # Blue
    elif "list" in b_type:
        return "rgb(0, 160, 0)"       # Green
    elif "key_value_pair" in b_type:
        return "rgb(100, 50, 150)"    # Deep Lavender / Purple
    else:
        return "rgb(110, 130, 0)"     # Olive default


def render_json_tree(data: Any, path: list = None,
                     blocks_lookup: dict = None,
                     max_depth: int = 0,
                     current_depth: int = 0) -> html.Div:
    """Generates a high-performance JSON tree view using collapsible details elements for on-demand lazy loading.

    Args:
        data (Any): Nested dictionary or list payload.
        path (list, optional): Key path tracking recursion depth. Defaults to None.
        blocks_lookup (dict, optional): Lookup mapping paths to raw blocks. Defaults to None.
        max_depth (int, optional): Maximum rendering depth (0 = top-level only). Defaults to 0.
        current_depth (int, optional): Current recursion tracking depth. Defaults to 0.

    Returns:
        html.Div: Rendered top-level HTML tree structure.
    """
    if path is None:
        path = []
    if blocks_lookup is None:
        blocks_lookup = {}

    # Helper to get background color and left border style for the handle badge box
    def get_grab_handle_style(current_path, key):
        block = blocks_lookup.get(tuple(current_path))
        if not block:
            block = blocks_lookup.get(tuple(key,))
        if block:
            b_type = block.get("type", "text")
            rgb_str = get_block_type_color(b_type)
            rgba_str = rgb_str.replace("rgb(", "rgba(").replace(")", ", 0.25)")
            return {"backgroundColor": rgba_str}
        return {"backgroundColor": "rgba(200, 200, 200, 0.25)"}

    if isinstance(data, dict):
        items = []
        for k, v in data.items():
            current_path = path + [k]
            path_str = json.dumps(current_path)
            input_key_str = f"json-key-{'--'.join([str(p) for p in current_path])}"

            key_input = html.Div(
                id=input_key_str,
                className="json-tree-key-container",
                children=[
                    dcc.Input(
                        id={'type': 'json-edit-key', 'path': path_str},
                        value=str(k),
                        debounce=True,
                        className="json-tree-input json-tree-key-input"
                    ),
                    dbc.Tooltip(
                        children=str(k),
                        target=input_key_str,
                        placement="top",
                        delay={"show": 300, "hide": 150},
                        style={"maxWidth": "200px", "wordWrap": "break-word"}
                    )
                ]
            )

            badge_style = get_grab_handle_style(current_path, k)

            if isinstance(v, (dict, list)):
                if current_depth >= max_depth:
                    items.append(
                        html.Details(
                            id={'type': 'json-lazy-container', 'path': path_str},
                            className="json-tree-branch json-tree-pill",
                            children=[
                                html.Summary(
                                    id={'type': 'json-lazy-summary', 'path': path_str},
                                    n_clicks=0,
                                    className="json-tree-summary",
                                    children=[
                                        # Left Zone: Colored Badge Box with Drag Handle
                                        html.Div(
                                            className="json-tree-badge",
                                            style=badge_style,
                                            children=[
                                                html.Span(className="json-tree-drag-handle", children="⋮⋮")
                                            ]
                                        ),
                                        # Center Zone: Key & Count info
                                        html.Div(
                                            className="json-tree-content-center",
                                            children=[
                                                key_input,
                                            ]
                                        ),
                                        # Right Zone: Caret is positioned automatically via CSS on <summary>
                                    ]
                                ),
                                html.Div(
                                    id={'type': 'json-lazy-content', 'path': path_str},
                                    className="json-tree-nested-content",
                                    children=[]
                                )
                            ],
                            open=False,
                        )
                    )
                else:
                    json_html_code = render_json_tree(
                        v, path=current_path, blocks_lookup=blocks_lookup,
                        max_depth=max_depth, current_depth=current_depth + 1
                    )
                    items.append(
                        html.Details(
                            className="json-tree-branch json-tree-pill",
                            children=[
                                html.Summary(
                                    className="json-tree-summary",
                                    children=[
                                        html.Div(
                                            className="json-tree-badge",
                                            style=badge_style,
                                            children=[
                                                html.Span(className="json-tree-drag-handle", children="⋮⋮")
                                            ]
                                        ),
                                        html.Div(
                                            className="json-tree-content-center",
                                            children=[
                                                key_input,
                                            ]
                                        ),
                                    ]
                                ),
                                html.Div(className="json-tree-nested-content", children=[json_html_code])
                            ],
                            open=False,
                        )
                    )
            else:
                input_val_str = f"json-val-{'--'.join([str(p) for p in current_path])}"
                items.append(
                    html.Div(
                        className="json-tree-leaf json-tree-pill",
                        children=[
                            # Left Zone: Colored Badge Box with Drag Handle
                            html.Div(
                                className="json-tree-badge",
                                style=badge_style,
                                children=[
                                    html.Span(className="json-tree-drag-handle", children="⋮⋮")
                                ]
                            ),
                            # Center Zone: Key, Colon, and Value Input
                            html.Div(
                                className="json-tree-content-center",
                                children=[
                                    key_input,
                                    html.Span(children=[": "], className="json-tree-colon"),
                                    html.Div(
                                        id=input_val_str,
                                        style={"display": "inline-block", "flexGrow": 1},
                                        children=[
                                            dcc.Input(
                                                id={'type': 'json-edit-value', 'path': path_str},
                                                value=str(v),
                                                debounce=True,
                                                className="json-tree-input json-tree-value-input"
                                            ),
                                            dbc.Tooltip(
                                                children=str(v),
                                                target=input_val_str,
                                                placement="top",
                                                delay={"show": 300, "hide": 150},
                                                style={"maxWidth": "200px", "wordWrap": "break-word"}
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                )
        return html.Div(items, className="json-tree-list")

    elif isinstance(data, list):
        items = []
        for idx, item in enumerate(data):
            current_path = path + [idx]
            path_str = json.dumps(current_path)
            input_val_str = f"json-val-{'--'.join([str(p) for p in current_path])}"
            badge_style = get_grab_handle_style(current_path, idx)

            if isinstance(item, (dict, list)):
                item_count = len(item)
                if current_depth >= max_depth:
                    items.append(
                        html.Details(
                            id={'type': 'json-lazy-container', 'path': path_str},
                            className="json-tree-branch json-tree-pill",
                            children=[
                                html.Summary(
                                    id={'type': 'json-lazy-summary', 'path': path_str},
                                    n_clicks=0,
                                    className="json-tree-summary",
                                    children=[
                                        html.Div(
                                            className="json-tree-badge",
                                            style=badge_style,
                                            children=[
                                                html.Span(className="json-tree-drag-handle", children="⋮⋮")
                                            ]
                                        ),
                                        html.Div(
                                            className="json-tree-content-center",
                                           children=[
                                                html.Span(children=[f"[{idx}]"], className="json-tree-index"),
                                            ]
                                        ),
                                    ]
                                ),
                                html.Div(
                                    id={'type': 'json-lazy-content', 'path': path_str},
                                    className="json-tree-nested-content",
                                    children=[]
                                )
                            ],
                            open=False,
                        )
                    )
                else:
                    json_html_code = render_json_tree(
                        item, path=current_path, blocks_lookup=blocks_lookup,
                        max_depth=max_depth, current_depth=current_depth + 1
                    )
                    items.append(
                        html.Details(
                            className="json-tree-branch json-tree-pill",
                            children=[
                                html.Summary(
                                    className="json-tree-summary",
                                    children=[
                                        html.Div(
                                            className="json-tree-badge",
                                            style=badge_style,
                                            children=[
                                                html.Span(className="json-tree-drag-handle", children="⋮⋮")
                                            ]
                                        ),
                                        html.Div(
                                            className="json-tree-content-center",
                                            children=[
                                                html.Span(children=[f"[{idx}]"], className="json-tree-index"),
                                            ]
                                        ),
                                    ]
                                ),
                                html.Div(className="json-tree-nested-content", children=[json_html_code])
                            ],
                            open=False,
                        )
                    )
            else:
                items.append(
                    html.Div(
                        className="json-tree-leaf json-tree-pill",
                        children=[
                            html.Div(
                                className="json-tree-badge",
                                style=badge_style,
                                children=[
                                    html.Span(className="json-tree-drag-handle", children="⋮⋮")
                                ]
                            ),
                            html.Div(
                                className="json-tree-content-center",
                                children=[
                                    html.Span(children=[f"[{idx}]"], className="json-tree-index"),
                                    html.Span(children=[": "], className="json-tree-colon"),
                                    html.Div(
                                        id=input_val_str,
                                        style={"display": "inline-block", "flexGrow": 1},
                                        children=[
                                            dcc.Input(
                                                id={'type': 'json-edit-value', 'path': path_str},
                                                value=str(item),
                                                debounce=True,
                                                className="json-tree-input json-tree-value-input"
                                            ),
                                            dbc.Tooltip(
                                                children=str(item),
                                                target={'type': 'json-edit-value', 'path': path_str},
                                                placement="top",
                                                delay={"show": 300, "hide": 150},
                                                style={"maxWidth": "200px", "wordWrap": "break-word"}
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                )
        return html.Div(items, className="json-tree-list")
    else:
        return html.Div(str(data), className="json-tree-scalar")


def debug_print(debug_str: str | list):
    """Appends single debug strings or lists of log messages to disk,
    safely creating the target directory if it doesn't exist.
    """
    # Anchor relative to this file's directory or use an absolute path,
    # e.g., placing results inside the studio project root or user home directory
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(log_dir, exist_ok=True)

    debug_file_path = os.path.join(log_dir, 'debug.txt')

    with open(debug_file_path, 'a', encoding='utf-8') as f:
        if isinstance(debug_str, str):
            print(debug_str, file=f)
        else:
            f.write('\n'.join(debug_str) + '\n')


def main():
    """ Main entry point for running the OGRRE Extract Studio web application."""
    app.run(debug=True, port=8050)

if __name__ == "__main__":
    main()