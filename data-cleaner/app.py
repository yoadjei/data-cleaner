"""
Data Cleaner - Professional Web Interface
Clean, modern UI with comprehensive error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO

# Page config
st.set_page_config(
    page_title="Data Cleaner",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize theme state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Theme toggle in top right
col_spacer, col_toggle = st.columns([10, 1])
with col_toggle:
    if st.button("◐" if st.session_state.theme == "dark" else "◑", help="Toggle theme"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

# Theme variables
if st.session_state.theme == "dark":
    theme = {
        "bg": "#0a0a0f",
        "card_bg": "#111116",
        "card_border": "rgba(255,255,255,0.06)",
        "text_primary": "#ffffff",
        "text_secondary": "#6b7280",
        "text_muted": "#a1a1aa",
        "input_bg": "#18181b",
        "input_border": "rgba(255,255,255,0.1)",
        "hover_bg": "#27272a",
        "upload_border": "rgba(255,255,255,0.1)",
        "upload_bg": "rgba(255,255,255,0.02)",
        "tab_bg": "#111116",
        "tab_selected": "#18181b",
        "alert_warning_bg": "rgba(245, 158, 11, 0.1)",
        "alert_warning_border": "rgba(245, 158, 11, 0.3)",
        "alert_warning_text": "#fbbf24",
        "log_border": "rgba(255,255,255,0.03)",
    }
else:
    theme = {
        "bg": "#ffffff",
        "card_bg": "#f9fafb",
        "card_border": "rgba(0,0,0,0.08)",
        "text_primary": "#111827",
        "text_secondary": "#6b7280",
        "text_muted": "#9ca3af",
        "input_bg": "#f3f4f6",
        "input_border": "rgba(0,0,0,0.1)",
        "hover_bg": "#e5e7eb",
        "upload_border": "rgba(0,0,0,0.15)",
        "upload_bg": "rgba(0,0,0,0.02)",
        "tab_bg": "#f3f4f6",
        "tab_selected": "#ffffff",
        "alert_warning_bg": "rgba(245, 158, 11, 0.08)",
        "alert_warning_border": "rgba(245, 158, 11, 0.4)",
        "alert_warning_text": "#b45309",
        "log_border": "rgba(0,0,0,0.06)",
    }

# Dynamic CSS based on theme
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    .stApp {{
        background: {theme["bg"]};
    }}
    
    /* Header */
    .header-container {{
        padding: 1rem 0 1rem 0;
        border-bottom: 1px solid {theme["card_border"]};
        margin-bottom: 2rem;
    }}
    
    .header-title {{
        font-size: 1.75rem;
        font-weight: 600;
        color: {theme["text_primary"]};
        letter-spacing: -0.02em;
        margin: 0;
    }}
    
    .header-subtitle {{
        font-size: 0.875rem;
        color: {theme["text_secondary"]};
        margin-top: 0.25rem;
        font-weight: 400;
    }}
    
    /* Cards */
    .card {{
        background: {theme["card_bg"]};
        border: 1px solid {theme["card_border"]};
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }}
    
    .card-header {{
        font-size: 0.75rem;
        font-weight: 500;
        color: {theme["text_secondary"]};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }}
    
    /* Stats */
    .stat-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
    }}
    
    .stat-item {{
        background: {theme["input_bg"]};
        border: 1px solid {theme["card_border"]};
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }}
    
    .stat-value {{
        font-size: 1.5rem;
        font-weight: 600;
        color: {theme["text_primary"]};
    }}
    
    .stat-label {{
        font-size: 0.75rem;
        color: {theme["text_secondary"]};
        margin-top: 0.25rem;
    }}
    
    /* Upload zone */
    .upload-zone {{
        border: 2px dashed {theme["upload_border"]};
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        background: {theme["upload_bg"]};
        transition: all 0.2s ease;
    }}
    
    .upload-zone:hover {{
        border-color: rgba(99, 102, 241, 0.5);
        background: rgba(99, 102, 241, 0.05);
    }}
    
    .upload-title {{
        font-size: 1rem;
        font-weight: 500;
        color: {theme["text_primary"]};
        margin-bottom: 0.5rem;
    }}
    
    .upload-hint {{
        font-size: 0.8rem;
        color: {theme["text_secondary"]};
    }}
    
    /* Table styling */
    .dataframe {{
        font-size: 0.8rem !important;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }}
    
    /* Download buttons */
    .stDownloadButton > button {{
        background: {theme["input_bg"]};
        border: 1px solid {theme["input_border"]};
        color: {theme["text_primary"]};
    }}
    
    .stDownloadButton > button:hover {{
        background: {theme["hover_bg"]};
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: {theme["tab_bg"]};
        border-radius: 8px;
        padding: 4px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-size: 0.8rem;
        font-weight: 500;
        color: {theme["text_secondary"]};
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {theme["tab_selected"]};
        color: {theme["text_primary"]};
    }}
    
    /* Select boxes */
    .stSelectbox > div > div {{
        background: {theme["input_bg"]};
        border-color: {theme["input_border"]};
    }}
    
    /* Alerts */
    .alert {{
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 0.8rem;
        margin: 0.5rem 0;
    }}
    
    .alert-warning {{
        background: {theme["alert_warning_bg"]};
        border: 1px solid {theme["alert_warning_border"]};
        color: {theme["alert_warning_text"]};
    }}
    
    .alert-success {{
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #22c55e;
    }}
    
    .alert-info {{
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #6366f1;
    }}
    
    /* Section titles */
    .section-title {{
        font-size: 0.875rem;
        font-weight: 600;
        color: {theme["text_primary"]};
        margin-bottom: 1rem;
    }}
    
    /* Log entries */
    .log-entry {{
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.75rem;
        color: {theme["text_muted"]};
        padding: 0.25rem 0;
        border-bottom: 1px solid {theme["log_border"]};
    }}
    
    /* Feature list */
    .feature-list {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin-top: 1.5rem;
    }}
    
    .feature-item {{
        background: {theme["card_bg"]};
        border: 1px solid {theme["card_border"]};
        border-radius: 8px;
        padding: 1rem;
    }}
    
    .feature-name {{
        font-size: 0.8rem;
        font-weight: 500;
        color: {theme["text_primary"]};
    }}
    
    .feature-desc {{
        font-size: 0.7rem;
        color: {theme["text_secondary"]};
        margin-top: 0.25rem;
    }}
    
    /* Hide Streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Expander */
    .streamlit-expanderHeader {{
        font-size: 0.8rem;
        font-weight: 500;
        color: {theme["text_primary"]};
    }}
    
    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {theme["text_primary"]};
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {theme["text_secondary"]};
    }}
    
    /* Mobile Responsive Styles */
    @media (max-width: 768px) {{
        .header-title {{
            font-size: 1.25rem;
        }}
        
        .header-subtitle {{
            font-size: 0.75rem;
        }}
        
        .upload-zone {{
            padding: 2rem 1rem;
        }}
        
        .upload-title {{
            font-size: 0.9rem;
        }}
        
        .feature-list {{
            grid-template-columns: 1fr;
            gap: 0.5rem;
        }}
        
        .feature-item {{
            padding: 0.75rem;
        }}
        
        .feature-name {{
            font-size: 0.75rem;
        }}
        
        .feature-desc {{
            font-size: 0.65rem;
        }}
        
        .section-title {{
            font-size: 0.8rem;
        }}
        
        .alert {{
            font-size: 0.7rem;
            padding: 0.5rem 0.75rem;
        }}
        
        .log-entry {{
            font-size: 0.65rem;
        }}
        
        /* Stack Streamlit columns on mobile */
        [data-testid="column"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
        }}
        
        /* Larger touch targets for buttons */
        .stButton > button {{
            padding: 1rem;
            font-size: 0.9rem;
            min-height: 48px;
        }}
        
        .stDownloadButton > button {{
            padding: 0.875rem;
            min-height: 48px;
        }}
        
        /* Smaller metrics on mobile */
        [data-testid="stMetricValue"] {{
            font-size: 1.25rem !important;
        }}
        
        [data-testid="stMetricLabel"] {{
            font-size: 0.7rem !important;
        }}
        
        /* Tab scrolling on mobile */
        .stTabs [data-baseweb="tab-list"] {{
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
        }}
        
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{
            display: none;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            padding: 0.5rem 0.75rem;
            font-size: 0.7rem;
            white-space: nowrap;
        }}
        
        /* Data table horizontal scroll */
        [data-testid="stDataFrame"] {{
            overflow-x: auto !important;
        }}
    }}
    
    /* Tablet adjustments */
    @media (max-width: 1024px) and (min-width: 769px) {{
        .feature-list {{
            grid-template-columns: repeat(2, 1fr);
        }}
    }}
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---

def sanitize_column_name(name):
    """Sanitize column names for safe display and processing"""
    if not isinstance(name, str):
        name = str(name)
    # Truncate long names
    if len(name) > 50:
        name = name[:47] + "..."
    return name


def truncate_for_display(name, max_len=30):
    """Truncate string for UI display"""
    if not isinstance(name, str):
        name = str(name)
    if len(name) > max_len:
        return name[:max_len-3] + "..."
    return name


def safe_numeric_convert(series):
    """Convert to numeric, handling commas and currency symbols"""
    if series.dtype == 'object':
        # Remove common formatting characters
        cleaned = series.astype(str).str.replace(r'[$€£¥,\s]', '', regex=True)
        # Handle parentheses as negative numbers
        cleaned = cleaned.str.replace(r'\(([0-9.]+)\)', r'-\1', regex=True)
        return pd.to_numeric(cleaned, errors='coerce')
    return pd.to_numeric(series, errors='coerce')


def detect_outliers_iqr(series):
    """Detect outliers using IQR method"""
    try:
        if not np.issubdtype(series.dtype, np.number):
            return pd.Series([False] * len(series), index=series.index)
        
        # Handle empty or all-NaN series
        valid_data = series.dropna()
        if len(valid_data) < 4:  # Need enough data for IQR
            return pd.Series([False] * len(series), index=series.index)
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        # Handle zero IQR (all same values)
        if IQR == 0:
            return pd.Series([False] * len(series), index=series.index)
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return (series < lower) | (series > upper)
    except Exception:
        return pd.Series([False] * len(series), index=series.index)


def safe_mean(series):
    """Calculate mean with NaN handling"""
    try:
        result = series.mean()
        return result if pd.notna(result) else 0
    except Exception:
        return 0


def safe_median(series):
    """Calculate median with NaN handling"""
    try:
        result = series.median()
        return result if pd.notna(result) else 0
    except Exception:
        return 0


def smart_impute(df, column, method):
    """Impute missing values with various methods - robust version"""
    try:
        series = df[column]
        
        # Check if all values are missing
        if series.isna().all():
            if method in ["mean", "median", "knn"]:
                return series.fillna(0)
            else:
                return series.fillna("UNKNOWN")
        
        if method == "mean":
            fill_val = safe_mean(series)
            return series.fillna(fill_val)
        
        elif method == "median":
            fill_val = safe_median(series)
            return series.fillna(fill_val)
        
        elif method == "mode":
            mode_val = series.mode()
            if len(mode_val) > 0:
                return series.fillna(mode_val.iloc[0])
            return series.fillna("UNKNOWN" if series.dtype == 'object' else 0)
        
        elif method == "knn":
            try:
                from sklearn.impute import KNNImputer
                if np.issubdtype(series.dtype, np.number):
                    non_null_count = series.notna().sum()
                    if non_null_count == 0:
                        return series.fillna(0)
                    if non_null_count < 2:
                        return series.fillna(safe_median(series))
                    
                    n_neighbors = min(5, non_null_count - 1)
                    n_neighbors = max(1, n_neighbors)
                    
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                    values = series.values.reshape(-1, 1)
                    imputed = imputer.fit_transform(values)
                    return pd.Series(imputed.flatten(), index=df.index)
                else:
                    mode_val = series.mode()
                    return series.fillna(mode_val.iloc[0] if len(mode_val) > 0 else "UNKNOWN")
            except ImportError:
                return series.fillna(safe_median(series))
            except Exception:
                return series.fillna(safe_median(series) if np.issubdtype(series.dtype, np.number) else "UNKNOWN")
        
        elif method == "drop":
            return series
        
        else:
            return series
    
    except Exception:
        # Ultimate fallback
        return df[column].fillna(0 if np.issubdtype(df[column].dtype, np.number) else "UNKNOWN")


def clean_dataframe(df, config):
    """Main cleaning function with comprehensive error handling"""
    log = []
    stats = {
        "original_rows": len(df),
        "duplicates_removed": 0,
        "missing_filled": 0,
        "outliers_found": 0,
        "final_rows": len(df)
    }
    
    try:
        df = df.copy()
        
        # Remove duplicates
        before = len(df)
        df = df.drop_duplicates()
        stats["duplicates_removed"] = before - len(df)
        if stats["duplicates_removed"] > 0:
            log.append(f"Removed {stats['duplicates_removed']} duplicate rows")
        
        # Process each column
        for col, settings in config.items():
            if col not in df.columns:
                continue
            
            try:
                # Type conversion
                if settings.get("convert_to"):
                    target = settings["convert_to"]
                    try:
                        if target in ["int", "float", "int64", "float64"]:
                            df[col] = safe_numeric_convert(df[col])
                        elif target == "string":
                            df[col] = df[col].astype(str)
                        log.append(f"Converted '{truncate_for_display(col)}' to {target}")
                    except Exception as e:
                        log.append(f"Failed to convert '{truncate_for_display(col)}': {str(e)[:50]}")
                
                # Imputation
                if settings.get("impute"):
                    missing_before = df[col].isnull().sum()
                    if missing_before > 0:
                        if settings["impute"] == "drop":
                            df = df.dropna(subset=[col])
                            log.append(f"Dropped {missing_before} rows with missing '{truncate_for_display(col)}'")
                        else:
                            df[col] = smart_impute(df, col, settings["impute"])
                            stats["missing_filled"] += missing_before
                            log.append(f"Filled {missing_before} missing in '{truncate_for_display(col)}' using {settings['impute']}")
                
                # Normalize strings
                if settings.get("normalize") and df[col].dtype == "object":
                    df[col] = df[col].astype(str).str.strip().str.lower()
                    log.append(f"Normalized text in '{truncate_for_display(col)}'")
            
            except Exception as e:
                log.append(f"Error processing '{truncate_for_display(col)}': {str(e)[:50]}")
                continue
        
        stats["final_rows"] = len(df)
        return df, log, stats
    
    except Exception as e:
        log.append(f"Critical error during cleaning: {str(e)[:100]}")
        stats["final_rows"] = len(df)
        return df, log, stats


def get_outlier_summary(df):
    """Get outlier counts for numeric columns - with error handling"""
    outliers = {}
    try:
        for col in df.select_dtypes(include=[np.number]).columns:
            try:
                outlier_mask = detect_outliers_iqr(df[col])
                count = outlier_mask.sum()
                if count > 0:
                    outliers[sanitize_column_name(col)] = count
            except Exception:
                continue
    except Exception:
        pass
    return outliers


def safe_div(a, b, default=0):
    """Safe division to avoid division by zero"""
    try:
        if b == 0:
            return default
        return a / b
    except Exception:
        return default


# --- Main App ---

# Header
st.markdown(f"""
<div class="header-container">
    <h1 class="header-title">Data Cleaner</h1>
    <p class="header-subtitle">Upload, analyze, and clean your datasets</p>
</div>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        # File size warning
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 50:
            st.warning(f"Large file ({file_size_mb:.1f} MB) - processing may be slow")
        
        # Try multiple encodings
        df = None
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        read_error = None
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='skip')
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                read_error = str(e)
                continue
        
        if df is None:
            st.error(f"Could not read file. {read_error or 'Unknown encoding issue.'}")
            st.stop()
        
        # Validate data
        if df.empty:
            st.error("The file is empty or contains no valid data.")
            st.stop()
        
        if len(df.columns) == 0:
            st.error("No columns detected.")
            st.stop()
        
        if len(df) == 0:
            st.error("No rows detected.")
            st.stop()
        
        # Sanitize column names for display
        display_columns = [truncate_for_display(col, 25) for col in df.columns]
        
        # Handle duplicate column names
        if len(df.columns) != len(set(df.columns)):
            st.warning("Duplicate column names detected. They have been auto-renamed.")
        
        # Single row warning
        if len(df) == 1:
            st.info("Single row detected - some statistics may be limited.")
        
        # Stats bar
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        with col4:
            st.metric("Duplicates", df.duplicated().sum())
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Two-column layout
        left_col, right_col = st.columns([1.2, 1])
        
        with left_col:
            st.markdown('<p class="section-title">Data Preview</p>', unsafe_allow_html=True)
            st.dataframe(df.head(8), use_container_width=True, height=300)
        
        with right_col:
            st.markdown('<p class="section-title">Column Analysis</p>', unsafe_allow_html=True)
            
            row_count = max(len(df), 1)  # Avoid division by zero
            analysis_df = pd.DataFrame({
                "Type": df.dtypes.astype(str),
                "Missing": df.isnull().sum(),
                "Missing %": (safe_div(df.isnull().sum(), row_count) * 100).round(1).astype(str) + "%"
            })
            st.dataframe(analysis_df, use_container_width=True, height=300)
        
        # Outlier detection
        outliers = get_outlier_summary(df)
        if outliers:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-title">Outliers Detected</p>', unsafe_allow_html=True)
            outlier_cols = st.columns(min(len(outliers), 4))
            for i, (col, count) in enumerate(outliers.items()):
                with outlier_cols[i % 4]:
                    st.markdown(f"""
                    <div class="alert alert-warning">
                        <strong>{truncate_for_display(col, 20)}</strong>: {count} outliers
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Configuration
        st.markdown('<p class="section-title">Cleaning Configuration</p>', unsafe_allow_html=True)
        
        config = {}
        
        # Limit tabs to prevent UI overflow
        max_tabs = 20
        columns_to_show = list(df.columns)[:max_tabs]
        if len(df.columns) > max_tabs:
            st.info(f"Showing first {max_tabs} columns. File has {len(df.columns)} total columns.")
        
        tabs = st.tabs([truncate_for_display(str(c), 15) for c in columns_to_show])
        
        for i, col in enumerate(columns_to_show):
            with tabs[i]:
                c1, c2, c3 = st.columns(3)
                
                col_config = {}
                
                with c1:
                    impute = st.selectbox(
                        "Missing values",
                        ["none", "mean", "median", "mode", "knn", "drop"],
                        key=f"impute_{i}_{col}"
                    )
                    if impute != "none":
                        col_config["impute"] = impute
                
                with c2:
                    convert = st.selectbox(
                        "Convert type",
                        ["keep", "int", "float", "string"],
                        key=f"type_{i}_{col}"
                    )
                    if convert != "keep":
                        col_config["convert_to"] = convert
                
                with c3:
                    if df[col].dtype == "object":
                        normalize = st.checkbox("Normalize text", key=f"norm_{i}_{col}")
                        if normalize:
                            col_config["normalize"] = True
                
                if col_config:
                    config[col] = col_config
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Clean button
        if st.button("Clean Data", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                cleaned_df, log, stats = clean_dataframe(df, config)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Results
            st.markdown('<p class="section-title">Results</p>', unsafe_allow_html=True)
            
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Original", f"{stats['original_rows']:,}")
            with r2:
                st.metric("Final", f"{stats['final_rows']:,}")
            with r3:
                st.metric("Removed", stats['duplicates_removed'])
            with r4:
                st.metric("Filled", stats['missing_filled'])
            
            # Log
            if log:
                with st.expander("Cleaning Log"):
                    for entry in log:
                        st.markdown(f'<div class="log-entry">{entry}</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Cleaned preview
            st.markdown('<p class="section-title">Cleaned Data</p>', unsafe_allow_html=True)
            st.dataframe(cleaned_df.head(8), use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Downloads
            d1, d2, d3 = st.columns(3)
            
            with d1:
                try:
                    csv_buffer = StringIO()
                    cleaned_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "Download CSV",
                        csv_buffer.getvalue(),
                        "cleaned_data.csv",
                        "text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error preparing CSV: {e}")
            
            with d2:
                st.download_button(
                    "Download Log",
                    "\n".join(log) if log else "No operations performed",
                    "cleaning_log.txt",
                    "text/plain",
                    use_container_width=True
                )
            
            with d3:
                try:
                    from ydata_profiling import ProfileReport
                    if st.button("Generate Report", use_container_width=True):
                        with st.spinner("Generating..."):
                            profile = ProfileReport(cleaned_df, minimal=True)
                            st.download_button(
                                "Download Report",
                                profile.to_html(),
                                "report.html",
                                "text/html"
                            )
                except ImportError:
                    st.info("Install ydata-profiling for reports")
                except Exception as e:
                    st.error(f"Report error: {str(e)[:50]}")

            st.session_state["cleaned_df"] = cleaned_df
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please check your file format and try again.")

else:
    # Welcome state
    st.markdown(f"""
    <div class="upload-zone">
        <p class="upload-title">Upload a CSV file to begin</p>
        <p class="upload-hint">Drag and drop or click to browse</p>
    </div>
    
    <div class="feature-list">
        <div class="feature-item">
            <p class="feature-name">Duplicate Removal</p>
            <p class="feature-desc">Automatically detect and remove duplicate rows</p>
        </div>
        <div class="feature-item">
            <p class="feature-name">Missing Value Handling</p>
            <p class="feature-desc">Mean, median, mode, or KNN imputation</p>
        </div>
        <div class="feature-item">
            <p class="feature-name">Outlier Detection</p>
            <p class="feature-desc">IQR-based statistical outlier identification</p>
        </div>
        <div class="feature-item">
            <p class="feature-name">Type Conversion</p>
            <p class="feature-desc">Convert columns to appropriate data types</p>
        </div>
        <div class="feature-item">
            <p class="feature-name">Text Normalization</p>
            <p class="feature-desc">Trim whitespace and standardize case</p>
        </div>
        <div class="feature-item">
            <p class="feature-name">Quality Reports</p>
            <p class="feature-desc">Generate comprehensive data profiles</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
