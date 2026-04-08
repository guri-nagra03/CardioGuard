"""
Data Ingestion Module

Handles CSV file uploads and provides intelligent sampling for longitudinal data.

UPDATED: Now samples 500 users × 30 days for longitudinal analysis
"""

import time
from pathlib import Path
from typing import Generator, Optional

import pandas as pd
import numpy as np

from config.settings import settings
from src.utils.constants import REQUIRED_COLUMNS
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class DataIngestionError(Exception):
    """Raised when data ingestion fails"""
    pass


def load_csv(
    filepath: Optional[str] = None,
    limit: Optional[int] = None,
    skip_validation: bool = False,
    n_users: int = 500,
    days_per_user: int = 30,
    random_state: int = 42,
    use_smart_sampling: bool = True,
    clean_before_sampling: bool = None
) -> pd.DataFrame:
    """
    Load wearable data from CSV file with intelligent sampling.

    TWO MODES:
    1. Legacy mode (limit parameter): Load first N rows
    2. Smart sampling (default): Sample N users × M days randomly
    
    Smart sampling strategy:
    - Optionally CLEAN sensor faults FIRST (recommended)
    - Randomly select n_users (default 500) to avoid selection bias
    - Keep last days_per_user (default 30) per user for longitudinal data
    - Results in ~15,000 daily records with 30-day trends per patient
    
    Args:
        filepath: Path to CSV file (default: settings.DATASET_PATH)
        limit: If provided, use legacy mode (first N rows). If None, use smart sampling.
        skip_validation: Skip schema validation (default: False)
        n_users: Number of users to sample in smart mode (default: 500)
        days_per_user: Days of history per user in smart mode (default: 30)
        random_state: Random seed for reproducibility (default: 42)
        use_smart_sampling: Enable smart sampling (default: True)
        clean_before_sampling: Remove sensor faults before sampling.
                               If None, use settings.CLEAN_BEFORE_SAMPLING (default: True)
                               This ensures training on genuine data, not imputed faults.

    Returns:
        DataFrame with wearable data

    Raises:
        DataIngestionError: If file not found or validation fails
        
    Examples:
        >>> # Smart sampling with sensor fault removal (RECOMMENDED)
        >>> df = load_csv()
        
        >>> # Smart sampling without cleaning first (legacy behavior)
        >>> df = load_csv(clean_before_sampling=False)
        
        >>> # Legacy mode: first 10,000 rows
        >>> df = load_csv(limit=10000)
        
        >>> # Custom sampling: 300 users × 60 days
        >>> df = load_csv(n_users=300, days_per_user=60)
    """
    filepath = filepath or settings.DATASET_PATH
    
    # Use settings default if not specified
    if clean_before_sampling is None:
        clean_before_sampling = settings.CLEAN_BEFORE_SAMPLING

    logger.info(f"Loading data from {filepath}")
    
    # Check if file exists
    if not Path(filepath).exists():
        raise DataIngestionError(f"Dataset not found at {filepath}")

    try:
        # Determine mode based on parameters
        if limit is not None:
            # LEGACY MODE: Simple row limit
            logger.info(f"Using LEGACY mode: loading first {limit} rows")
            df = pd.read_csv(filepath, nrows=limit)
            logger.info(f"Loaded {len(df)} rows (legacy mode)")
            
        elif use_smart_sampling:
            # SMART SAMPLING MODE: Sample users and days
            logger.info(
                f"Using SMART SAMPLING: {n_users} users × {days_per_user} days "
                f"(random_state={random_state})"
            )
            
            # Load full dataset (or large chunk to get user distribution)
            df_full = pd.read_csv(filepath)
            logger.info(f"Loaded full dataset: {len(df_full):,} rows")
            
            # OPTIONAL: Clean sensor faults BEFORE sampling
            if clean_before_sampling:
                logger.info("Cleaning sensor faults BEFORE sampling (recommended)...")
                # Import here to avoid circular dependency
                from src.data.preprocessing import remove_sensor_faults
                
                df_full = remove_sensor_faults(df_full)
                logger.info(f"After sensor fault removal: {len(df_full):,} rows")
            
            # Apply smart sampling
            df = _smart_sample_users_and_days(
                df_full,
                n_users=n_users,
                days_per_user=days_per_user,
                random_state=random_state
            )
            
            logger.info(
                f"Smart sampling complete: {len(df):,} rows from "
                f"{df['user_id'].nunique()} users"
            )
        else:
            # Load without sampling
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df):,} rows (no sampling)")

        # Validate schema unless skipped
        if not skip_validation:
            validate_schema(df)
            logger.info("Schema validation passed")

        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Sort by user_id and date
        df = df.sort_values(['user_id', 'date']).reset_index(drop=True)

        # Log final summary
        logger.info(
            f"Data ingestion complete: {len(df):,} rows, "
            f"{df['user_id'].nunique()} unique users"
        )
        
        # Log date range
        date_range_days = (df['date'].max() - df['date'].min()).days
        logger.info(
            f"Date range: {df['date'].min().date()} to {df['date'].max().date()} "
            f"({date_range_days} days)"
        )
        
        return df

    except pd.errors.EmptyDataError:
        raise DataIngestionError("CSV file is empty")
    except pd.errors.ParserError as e:
        raise DataIngestionError(f"Failed to parse CSV: {e}")
    except Exception as e:
        raise DataIngestionError(f"Unexpected error loading data: {e}")


def _smart_sample_users_and_days(
    df: pd.DataFrame,
    n_users: int = 500,
    days_per_user: int = 30,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Smart sampling: Randomly sample N users and keep last M days per user.
    
    This creates longitudinal data suitable for:
    - 30-day trend visualization
    - Time-series feature engineering
    - Realistic ML training scenarios
    
    Args:
        df: Full DataFrame
        n_users: Number of users to sample
        days_per_user: Days of history per user
        random_state: Random seed
        
    Returns:
        Sampled DataFrame
    """
    # Get all unique users
    all_users = df['user_id'].unique()
    total_users = len(all_users)
    
    logger.info(f"Total unique users in dataset: {total_users}")
    
    # Randomly sample users (avoid selection bias)
    np.random.seed(random_state)
    
    if n_users > total_users:
        logger.warning(
            f"Requested {n_users} users but only {total_users} available. "
            f"Using all users."
        )
        selected_users = all_users
    else:
        selected_users = np.random.choice(
            all_users,
            size=n_users,
            replace=False
        )
    
    logger.info(f"Randomly selected {len(selected_users)} users")
    
    # Filter to selected users
    df_selected = df[df['user_id'].isin(selected_users)].copy()
    logger.info(f"After user selection: {len(df_selected):,} rows")
    
    # Keep last N days per user
    def get_last_n_days(group):
        """Get last N days for a user, sorted by date"""
        return group.sort_values('date').tail(days_per_user)
    
    df_sampled = df_selected.groupby('user_id', group_keys=False).apply(
        get_last_n_days
    ).reset_index(drop=True)
    
    logger.info(
        f"After keeping last {days_per_user} days per user: {len(df_sampled):,} rows"
    )
    
    # Log rows per user statistics
    rows_per_user = df_sampled.groupby('user_id').size()
    logger.info(
        f"Rows per user: mean={rows_per_user.mean():.1f}, "
        f"min={rows_per_user.min()}, max={rows_per_user.max()}"
    )
    
    return df_sampled


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has required columns and correct types.

    Args:
        df: DataFrame to validate

    Returns:
        True if validation passes

    Raises:
        DataIngestionError: If validation fails
    """
    # Check required columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise DataIngestionError(
            f"Missing required columns: {missing_columns}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    # Check for empty DataFrame
    if df.empty:
        raise DataIngestionError("DataFrame is empty")

    # Validate numeric columns
    numeric_columns = [
        'steps', 'calories_burned', 'distance_km',
        'active_minutes', 'sleep_hours', 'heart_rate_avg'
    ]

    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Try to convert
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.warning(f"Converted column '{col}' to numeric type")
            except Exception as e:
                raise DataIngestionError(f"Column '{col}' must be numeric: {e}")

    # Validate date column
    try:
        pd.to_datetime(df['date'], errors='coerce')
    except Exception as e:
        raise DataIngestionError(f"Column 'date' must be in valid date format: {e}")

    # Check for all-null columns
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        raise DataIngestionError(f"Columns contain all null values: {null_columns}")

    logger.debug("Schema validation successful")
    return True


def select_quality_patients(
    df: pd.DataFrame,
    n_users: int = 500,
    min_days: int = 30,
    days_per_user: int = 30,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Select n_users patients who have sufficient clean data.

    Called AFTER preprocessing so that the selected patients are guaranteed
    to have at least min_days of valid, cleaned records.

    Args:
        df: Fully preprocessed DataFrame (all users, all dates)
        n_users: Number of patients to select
        min_days: Minimum clean days required to be eligible
        days_per_user: How many of the most-recent days to keep per patient
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with n_users patients, each with up to days_per_user rows
    """
    # Count clean days per user after preprocessing
    days_per_user_series = df.groupby('user_id').size()

    eligible_users = days_per_user_series[
        days_per_user_series >= min_days
    ].index.tolist()

    logger.info(
        f"Eligible patients (>= {min_days} clean days): "
        f"{len(eligible_users)} / {df['user_id'].nunique()}"
    )

    if len(eligible_users) == 0:
        raise DataIngestionError(
            f"No patients have >= {min_days} clean days. "
            f"Lower min_days or check preprocessing."
        )

    # Randomly sample from eligible patients
    rng = np.random.default_rng(random_state)
    if n_users >= len(eligible_users):
        logger.warning(
            f"Requested {n_users} patients but only {len(eligible_users)} eligible. "
            f"Using all eligible patients."
        )
        selected_users = eligible_users
    else:
        selected_users = rng.choice(
            eligible_users, size=n_users, replace=False
        ).tolist()

    logger.info(f"Selected {len(selected_users)} patients from eligible pool")

    # Keep the most-recent days_per_user days for each selected patient
    df_selected = df[df['user_id'].isin(selected_users)]
    df_final = (
        df_selected
        .groupby('user_id', group_keys=False)
        .apply(lambda g: g.sort_values('date').tail(days_per_user))
        .reset_index(drop=True)
    )

    rows_per_user = df_final.groupby('user_id').size()
    logger.info(
        f"Final dataset: {len(df_final):,} rows from {len(selected_users)} patients "
        f"(mean {rows_per_user.mean():.1f} days/patient)"
    )

    return df_final


def simulate_stream(
    dataframe: pd.DataFrame,
    batch_size: int = 100,
    delay_seconds: float = 5.0
) -> Generator[pd.DataFrame, None, None]:
    """
    Simulate real-time data streaming by yielding batches with delays.

    This is for educational demonstration of how streaming data ingestion
    would work with wearable devices sending data periodically.

    Args:
        dataframe: Source DataFrame to stream
        batch_size: Number of rows per batch
        delay_seconds: Delay between batches (simulates network latency)

    Yields:
        DataFrame batches

    Example:
        >>> df = load_csv()
        >>> for batch in simulate_stream(df, batch_size=50, delay_seconds=2):
        ...     process_batch(batch)
    """
    total_rows = len(dataframe)
    logger.info(
        f"Starting simulated stream: {total_rows} rows, "
        f"batch_size={batch_size}, delay={delay_seconds}s"
    )

    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch = dataframe.iloc[start_idx:end_idx].copy()

        logger.debug(f"Yielding batch: rows {start_idx} to {end_idx}")
        yield batch

        # Simulate network delay (skip delay for last batch)
        if end_idx < total_rows:
            time.sleep(delay_seconds)

    logger.info("Simulated stream complete")


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the dataset.

    Args:
        df: DataFrame to summarize

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_rows': len(df),
        'unique_users': df['user_id'].nunique(),
        'date_range': {
            'start': df['date'].min(),
            'end': df['date'].max(),
            'days': (df['date'].max() - df['date'].min()).days
        },
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df[[
            'steps', 'calories_burned', 'distance_km',
            'active_minutes', 'sleep_hours', 'heart_rate_avg'
        ]].describe().to_dict()
    }

    return summary


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter DataFrame by date range.

    Args:
        df: DataFrame to filter
        start_date: Start date (inclusive) in YYYY-MM-DD format
        end_date: End date (inclusive) in YYYY-MM-DD format

    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()

    if start_date:
        start = pd.to_datetime(start_date)
        filtered_df = filtered_df[filtered_df['date'] >= start]
        logger.info(f"Filtered to dates >= {start_date}: {len(filtered_df)} rows")

    if end_date:
        end = pd.to_datetime(end_date)
        filtered_df = filtered_df[filtered_df['date'] <= end]
        logger.info(f"Filtered to dates <= {end_date}: {len(filtered_df)} rows")

    return filtered_df


def filter_by_users(
    df: pd.DataFrame,
    user_ids: list[int]
) -> pd.DataFrame:
    """
    Filter DataFrame to specific user IDs.

    Args:
        df: DataFrame to filter
        user_ids: List of user IDs to include

    Returns:
        Filtered DataFrame
    """
    filtered_df = df[df['user_id'].isin(user_ids)].copy()
    logger.info(f"Filtered to {len(user_ids)} users: {len(filtered_df)} rows")
    return filtered_df


# Example usage
if __name__ == "__main__":
    print("=== CardioGuard Data Ingestion Demo ===\n")
    
    # Test smart sampling (default)
    print("1. Smart Sampling (500 users × 30 days):")
    print("-" * 60)
    df = load_csv()
    print(f"   Rows: {len(df):,}")
    print(f"   Users: {df['user_id'].nunique()}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Check rows per user
    rows_per_user = df.groupby('user_id').size()
    print(f"   Rows per user: mean={rows_per_user.mean():.1f}, min={rows_per_user.min()}, max={rows_per_user.max()}")
    
    # Test legacy mode
    print("\n2. Legacy Mode (first 10,000 rows):")
    print("-" * 60)
    df_legacy = load_csv(limit=10000)
    print(f"   Rows: {len(df_legacy):,}")
    print(f"   Users: {df_legacy['user_id'].nunique()}")
    
    # Get summary
    print("\n3. Dataset Summary:")
    print("-" * 60)
    summary = get_dataset_summary(df)
    print(f"   Total rows: {summary['total_rows']:,}")
    print(f"   Unique users: {summary['unique_users']}")
    print(f"   Date range: {summary['date_range']['start'].date()} to {summary['date_range']['end'].date()}")
    print(f"   Total days: {summary['date_range']['days']}")
    
    print("\n✓ Ingestion tests complete!")