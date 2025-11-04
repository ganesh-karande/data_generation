# streamlit_synthetic_text_hash_safe.py

import streamlit as st
import pandas as pd
import pickle
import os

# CTGAN for single table
from ctgan import CTGAN

# SDV for multi-table
from sdv.multi_table import HMASynthesizer
from sdv.metadata import MultiTableMetadata

st.title("🧪 Synthetic Data Generator")
st.write(
    "Upload CSV dataset(s) and generate synthetic data using CTGAN (single table) or HMASynthesizer (multi-table)."
)

# -----------------
# Upload CSV(s)
# -----------------
uploaded_files = st.file_uploader(
    "Upload CSV file(s) (for multi-table, upload multiple CSVs)", type="csv", accept_multiple_files=True
)

if uploaded_files:
    # Read all uploaded CSVs
    data_dict = {}
    for file in uploaded_files:
        df = pd.read_csv(file)
        data_dict[file.name.split(".")[0]] = df
        st.write(f"Preview of {file.name}:")
        st.dataframe(df.head())

    # Select model type
    mode = st.selectbox("Select Dataset Type", ["Single Table", "Multi Table"])

    # Number of synthetic rows
    rows_count = st.number_input("Number of synthetic rows to generate", min_value=1, value=1000)

    # Maximum text length for multi-table truncation
    max_text_len = st.number_input("Max characters for text columns (multi-table)", min_value=50, value=200)

    if st.button("Generate Synthetic Data"):
        os.makedirs("models", exist_ok=True)

        # -----------------
        # Single Table Mode
        # -----------------
        if mode == "Single Table":
            st.info("Preprocessing and training CTGAN on your dataset...")

            df = list(data_dict.values())[0]  # Take first CSV for single-table

            # Automatically hash long text columns for CTGAN compatibility
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].astype(str).apply(lambda x: hash(x))

            # Train CTGAN
            ctgan = CTGAN(epochs=200, verbose=True)
            ctgan.fit(df)

            # Save model
            with open("models/ctgan_single.pkl", "wb") as f:
                pickle.dump(ctgan, f)
            st.success("✅ CTGAN model trained and saved!")

            # Generate synthetic data
            synthetic_data = ctgan.sample(rows_count)
            st.write("Preview of Synthetic Data:")
            st.dataframe(synthetic_data.head())

            # Download button
            csv = synthetic_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Synthetic CSV",
                data=csv,
                file_name="synthetic_data.csv",
                mime="text/csv",
            )

        # -----------------
        # Multi Table Mode
        # -----------------
        elif mode == "Multi Table":
            st.info("Preprocessing and training HMASynthesizer on your dataset...")

            # Truncate long text columns in each table
            for table_name, df in data_dict.items():
                for col in df.select_dtypes(include=["object"]).columns:
                    df[col] = df[col].astype(str).str[:max_text_len]
                data_dict[table_name] = df

            # Build metadata automatically
            metadata = MultiTableMetadata()
            for table_name, df in data_dict.items():
                metadata.add_table(table_name, table_data=df)

            # Detect primary keys (first unique column per table)
            for table_name, df in data_dict.items():
                for col in df.columns:
                    if df[col].is_unique:
                        metadata.update_column(table_name, col, sdtype="id")
                        metadata.set_primary_key(table_name, col)
                        break

            # Train HMASynthesizer
            synthesizer = HMASynthesizer(metadata)
            synthesizer.fit(data_dict)

            # Save model
            with open("models/hma_synthesizer.pkl", "wb") as f:
                pickle.dump(synthesizer, f)
            st.success("✅ HMASynthesizer model trained and saved!")

            # Generate synthetic data
            synthetic_data = synthesizer.sample(
                scale=rows_count // len(next(iter(data_dict.values())))
            )

            # Show previews and download buttons
            for table_name, df in synthetic_data.items():
                st.write(f"Preview of Synthetic Data ({table_name}):")
                st.dataframe(df.head())
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 Download {table_name} CSV",
                    data=csv,
                    file_name=f"synthetic_{table_name}.csv",
                    mime="text/csv",
                )
