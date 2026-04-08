"""
Training Script

Complete end-to-end training pipeline for CardioGuard:
1. Load and preprocess fitness tracker data
2. Engineer cardiovascular features
3. Generate synthetic risk labels
4. Train ML model (logistic regression)
5. Generate predictions and stratifications for all patients
6. Populate SQLite cache
7. Optionally post to FHIR server

Usage:
    python scripts/train_model.py [--no-fhir] [--limit N]

Arguments:
    --no-fhir: Skip FHIR server operations (cache-only mode)
    --limit N: Limit to N rows (default: 10000)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from tqdm import tqdm

from config.settings import settings
from src.data.ingestion import load_csv, select_quality_patients
from src.data.preprocessing import clean_data
from src.data.feature_engineering import (
    create_cardiovascular_features,
    get_feature_columns
)
from src.ml.label_generator import generate_synthetic_labels
from src.ml.trainer import train_model, save_model
from src.ml.predictor import RiskPredictor
from src.ml.explainer import RiskExplainer
from src.risk.stratification import RiskStratifier
from src.storage.fhir_repository import FHIRRepository
from src.utils.logging_config import setup_logging
from src.utils.constants import LOINC_CODES, UNITS

logger = setup_logging(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CardioGuard ML model and populate database"
    )

    parser.add_argument(
        "--no-fhir",
        action="store_true",
        help="Skip FHIR server operations (cache-only mode)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Limit number of rows to process (default: 10000)"
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training (use existing model)"
    )

    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    print("=" * 80)
    print("CardioGuard Training Pipeline")
    print("=" * 80)
    print()

    # Configuration
    print("Configuration:")
<<<<<<< HEAD
    print(f"  Sampling: Smart (500 users x 30 days)")  # ← ADD THIS INSTEAD
    print(f"  Clean before sampling: {settings.CLEAN_BEFORE_SAMPLING}")
=======
    print(f"  Data limit: {args.limit} rows")
>>>>>>> 0fadd6eefaaaf69720688849a8adfa847f925c39
    print(f"  FHIR operations: {'Disabled' if args.no_fhir else 'Enabled'}")
    print(f"  Skip training: {'Yes' if args.skip_training else 'No'}")
    print()

    # =========================================================================
    # Step 1: Load and Preprocess ALL Data
    # =========================================================================
    print("Step 1: Loading and preprocessing data...")
    print("-" * 80)

    try:
        # Load the full dataset — no patient selection yet
        print("  Loading full dataset (no sampling)...")
        df_raw = load_csv(use_smart_sampling=False)
        print(f"✓ Loaded {len(df_raw):,} rows ({df_raw['user_id'].nunique()} patients)")

        # Clean ALL data before selecting patients
        print("  Cleaning full dataset...")
        df_clean_all = clean_data(df_raw)
        print(f"✓ Cleaned data: {len(df_clean_all):,} rows remaining")

        # Now select 500 patients from the cleaned data
        print("  Selecting 500 quality patients from cleaned data...")
        df_clean = select_quality_patients(
            df_clean_all,
            n_users=500,
            min_days=30,
            days_per_user=30
        )
        print(
            f"✓ Selected {df_clean['user_id'].nunique()} patients "
            f"({len(df_clean):,} rows, each with >= 30 clean days)"
        )

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        print(f"✗ Error loading data: {e}")
        print("\nMake sure the dataset exists at:")
        print(f"  {settings.DATASET_PATH}")
        return 1

    # =========================================================================
    # Step 2: Feature Engineering
    # =========================================================================
    print("\nStep 2: Engineering cardiovascular features...")
    print("-" * 80)

    try:
        features_df = create_cardiovascular_features(df_clean)
        print(f"✓ Created features for {len(features_df)} patient-days")

        feature_cols = get_feature_columns()
        print(f"✓ Feature columns: {len(feature_cols)}")
        for col in feature_cols:
            print(f"    - {col}")

        # Drop rows with missing features
        X = features_df[feature_cols].dropna()
        print(f"✓ Features ready: {len(X)} complete rows")

        if len(X) == 0:
            print("✗ No valid features found. Check data quality.")
            return 1

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        print(f"✗ Error creating features: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # =========================================================================
    # Step 3: Generate Synthetic Labels
    # =========================================================================
    print("\nStep 3: Generating synthetic risk labels...")
    print("-" * 80)

    try:
        # Align features_df with X (same indices)
        features_aligned = features_df.loc[X.index]

        y = generate_synthetic_labels(features_aligned)
        print(f"✓ Generated {len(y)} synthetic labels")

        # Label distribution
        label_counts = y.value_counts().sort_index()
        print("  Label distribution:")
        print(f"    Low Risk (0):    {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(y)*100:.1f}%)")
        print(f"    Medium Risk (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(y)*100:.1f}%)")
        print(f"    High Risk (2):   {label_counts.get(2, 0)} ({label_counts.get(2, 0)/len(y)*100:.1f}%)")

    except Exception as e:
        logger.error(f"Label generation failed: {e}")
        print(f"✗ Error generating labels: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # =========================================================================
    # Step 4: Train ML Model
    # =========================================================================
    if not args.skip_training:
        print("\nStep 4: Training ML model...")
        print("-" * 80)

        try:
            model, scaler, metrics = train_model(X, y)
            print(f"✓ Model trained successfully")

            print("  Model Performance:")
            print(f"    Accuracy:  {metrics.get('accuracy', float('nan')):.3f}")
            print(f"    Precision: {metrics.get('precision', float('nan')):.3f}")
            print(f"    Recall:    {metrics.get('recall', float('nan')):.3f}")
            print(f"    F1 Score:  {metrics.get('f1_score', float('nan')):.3f}")

            if 'cv_accuracy' in metrics:
                print(f"    CV Accuracy: {metrics['cv_accuracy']:.3f}")


            # Save model
            model_path = settings.MODEL_PATH
            scaler_path = settings.SCALER_PATH

            save_model(model, scaler, model_path, scaler_path)
            print(f"✓ Model saved to {model_path}")
            print(f"✓ Scaler saved to {scaler_path}")

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            print(f"✗ Error training model: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("\nStep 4: Skipping model training (using existing model)...")
        print("-" * 80)

    # =========================================================================
    # Step 5: Generate Predictions
    # =========================================================================
    print("\nStep 5: Generating predictions for all patients...")
    print("-" * 80)

    try:
        predictor = RiskPredictor()
        stratifier = RiskStratifier()

        # Get predictions
        predictions = predictor.predict_batch(X)
        print(f"✓ Generated {len(predictions)} predictions")

        # Risk score distribution
        import numpy as np
        print(f"  Risk score statistics:")
        print(f"    Mean: {predictions['risk_score'].mean():.3f}")
        print(f"    Median: {predictions['risk_score'].median():.3f}")
        print(f"    Min: {predictions['risk_score'].min():.3f}")
        print(f"    Max: {predictions['risk_score'].max():.3f}")

    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        print(f"✗ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # =========================================================================
    # Step 6: Initialize Storage
    # =========================================================================
    print("\nStep 6: Initializing storage layer...")
    print("-" * 80)

    try:
        repo = FHIRRepository(enable_fhir=not args.no_fhir)
        print(f"✓ Repository initialized")
        print(f"  FHIR server: {'Available' if repo.is_fhir_available() else 'Unavailable (cache-only mode)'}")

    except Exception as e:
        logger.error(f"Storage initialization failed: {e}")
        print(f"✗ Error initializing storage: {e}")
        return 1

    # =========================================================================
    # Step 7: Process Patients and Populate Database
    # =========================================================================
    print("\nStep 7: Processing patients and populating database...")
    print("-" * 80)

    try:
        # Clear existing data so re-runs don't accumulate duplicates
        repo.cache.clear_cache(confirm=True)
        print("✓ Cleared existing cache data")

        # Get unique patient IDs
        features_aligned = features_df.loc[X.index]
        patient_ids = features_aligned['user_id'].unique()

        print(f"Processing {len(patient_ids)} unique patients...")

        successful_patients = 0
        failed_patients = 0

        for patient_id in tqdm(patient_ids, desc="Processing patients"):
            try:
                # Get patient data
                patient_data = features_aligned[features_aligned['user_id'] == patient_id]

                if len(patient_data) == 0:
                    continue

                # Get latest features for this patient
                latest_features = patient_data.iloc[-1]
                latest_date = latest_features['date']

                # Get feature values for prediction
                feature_values = latest_features[feature_cols].to_dict()

                # Predict
                prediction = predictor.predict(latest_features[feature_cols])

                # Stratify
                stratification = stratifier.stratify(
                    ml_score=prediction['risk_score'],
                    features=feature_values,
                    patient_id=patient_id
                )

                # Collect observation metadata for ALL 30 days (always, regardless of FHIR)
                observations = []
                observation_metadata = []
                risk_assessment = None
                flag = None

                patient_raw_data = df_clean[df_clean['user_id'] == patient_id]

                if len(patient_raw_data) > 0:
                    for _, day_row in patient_raw_data.iterrows():
                        obs_date = day_row['date']

                        for metric_name in LOINC_CODES.keys():
                            if metric_name in day_row and pd.notna(day_row[metric_name]):
                                observation_metadata.append({
                                    'metric_name': metric_name,
                                    'value': float(day_row[metric_name]),
                                    'unit': UNITS.get(metric_name, ''),
                                    'date': str(obs_date)
                                })

                if not args.no_fhir:
                    # Lazy import: only load fhir.resources when FHIR is enabled
                    from src.fhir.converter import create_observation
                    from src.fhir.risk_resources import create_risk_assessment, create_risk_flag

                    if len(patient_raw_data) > 0:
                        for _, day_row in patient_raw_data.iterrows():
                            obs_date = day_row['date']

                            for metric_name in LOINC_CODES.keys():
                                if metric_name in day_row and pd.notna(day_row[metric_name]):
                                    obs = create_observation(
                                        user_id=patient_id,
                                        date=obs_date,
                                        metric_name=metric_name,
                                        value=day_row[metric_name]
                                    )
                                    observations.append(obs)

                    # Create RiskAssessment
                    risk_assessment = create_risk_assessment(
                        user_id=patient_id,
                        ml_score=prediction['risk_score'],
                        risk_level=stratification['risk_level']
                    )

                    # Create Flag (if applicable)
                    flag = create_risk_flag(
                        user_id=patient_id,
                        risk_level=stratification['risk_level'],
                        reason=f"Cardiovascular wellness risk: {stratification['risk_level']}"
                    )

                # Process patient through repository
                result = repo.process_patient(
                    patient_id=patient_id,
                    observations=observations,
                    observation_metadata=observation_metadata,
                    prediction=prediction,
                    stratification=stratification,
                    risk_assessment=risk_assessment,
                    flag=flag
                )

                if len(result.get('errors', [])) == 0:
                    successful_patients += 1
                else:
                    logger.warning(f"Patient {patient_id} processed with errors: {result['errors']}")
                    failed_patients += 1

            except Exception as e:
                logger.error(f"Failed to process patient {patient_id}: {e}")
                failed_patients += 1
                continue

        print(f"\n✓ Processing complete:")
        print(f"    Successful: {successful_patients}")
        print(f"    Failed: {failed_patients}")

    except Exception as e:
        logger.error(f"Patient processing failed: {e}")
        print(f"✗ Error processing patients: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # =========================================================================
    # Step 8: Summary Statistics
    # =========================================================================
    print("\nStep 8: Summary statistics...")
    print("-" * 80)

    try:
        stats = repo.get_cache_stats()

        print("Database Statistics:")
        print(f"  Total patients: {stats.get('total_patients', 0)}")
        print(f"  Total observations: {stats.get('total_observations', 0)}")
        print(f"  Total predictions: {stats.get('total_predictions', 0)}")
        print(f"  Total stratifications: {stats.get('total_stratifications', 0)}")

        risk_dist = stats.get('risk_distribution', {})
        if risk_dist:
            print("\n  Risk Distribution:")
            print(f"    Green (Low):    {risk_dist.get('Green', 0)}")
            print(f"    Yellow (Medium): {risk_dist.get('Yellow', 0)}")
            print(f"    Red (High):     {risk_dist.get('Red', 0)}")

    except Exception as e:
        logger.warning(f"Failed to get statistics: {e}")

    # =========================================================================
    # Completion
    # =========================================================================
    print("\n" + "=" * 80)
    print("Training Pipeline Complete!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Start the Streamlit UI:")
    print("     streamlit run ui/app.py")
    print()
    print("  2. Or start with Docker:")
    print("     docker-compose up")
    print()
    print("  3. Login with demo credentials:")
    print("     Username: clinician1")
    print("     Password: demo123")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
