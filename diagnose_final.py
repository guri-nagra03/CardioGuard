"""
Diagnostic Script: Check Data Distribution (FINAL FIX)

Uses correct table names: predictions and stratifications
"""

import pandas as pd
import numpy as np
import sqlite3

# Connect to your database
conn = sqlite3.connect('data/cache/cardioguard.db')

print("=== DATABASE TABLES ===")
print("✓ predictions")
print("✓ stratifications")

# Get predictions data
predictions = pd.read_sql("SELECT * FROM predictions", conn)
print(f"\n=== PREDICTIONS TABLE ({len(predictions)} records) ===")
print("Columns:", predictions.columns.tolist())
print(f"\nML Score stats:")
print(f"  Mean: {predictions['ml_score'].mean():.3f}")
print(f"  Median: {predictions['ml_score'].median():.3f}")
print(f"  Min: {predictions['ml_score'].min():.3f}")
print(f"  Max: {predictions['ml_score'].max():.3f}")

print("\nML Score distribution:")
print(f"  < 0.55 (should be Green): {(predictions['ml_score'] < 0.55).sum()} ({(predictions['ml_score'] < 0.55).mean()*100:.1f}%)")
print(f"  0.55-0.80 (should be Yellow): {((predictions['ml_score'] >= 0.55) & (predictions['ml_score'] < 0.80)).sum()} ({((predictions['ml_score'] >= 0.55) & (predictions['ml_score'] < 0.80)).mean()*100:.1f}%)")
print(f"  >= 0.80 (should be Red): {(predictions['ml_score'] >= 0.80).sum()} ({(predictions['ml_score'] >= 0.80).mean()*100:.1f}%)")

# Get stratifications data
stratifications = pd.read_sql("SELECT * FROM stratifications", conn)
print(f"\n=== STRATIFICATIONS TABLE ({len(stratifications)} records) ===")
print("Columns:", stratifications.columns.tolist())

if 'risk_level' in stratifications.columns:
    print("\nActual Risk Distribution:")
    print(stratifications['risk_level'].value_counts())
    print(f"\nPercentages:")
    for level in ['Green', 'Yellow', 'Red']:
        count = (stratifications['risk_level'] == level).sum()
        pct = count / len(stratifications) * 100
        print(f"  {level}: {count} ({pct:.1f}%)")

# Check for override information
if 'override_applied' in stratifications.columns:
    print("\n=== OVERRIDE ANALYSIS ===")
    overridden = stratifications[stratifications['override_applied'] == True]
    print(f"Patients with overrides: {len(overridden)} ({len(overridden)/len(stratifications)*100:.1f}%)")
    
    if 'override_reason' in stratifications.columns and len(overridden) > 0:
        print("\nOverride reasons:")
        print(overridden['override_reason'].value_counts())

# Merge predictions and stratifications
merged = predictions.merge(stratifications, on='patient_id', how='left')

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Calculate what distribution SHOULD be based on ML scores
ml_green = (merged['ml_score'] < 0.55).sum()
ml_yellow = ((merged['ml_score'] >= 0.55) & (merged['ml_score'] < 0.80)).sum()
ml_red = (merged['ml_score'] >= 0.80).sum()

actual_green = (merged['risk_level'] == 'Green').sum()
actual_yellow = (merged['risk_level'] == 'Yellow').sum()
actual_red = (merged['risk_level'] == 'Red').sum()

print(f"\nBased on ML scores alone (before overrides):")
print(f"  Green: {ml_green} ({ml_green/len(merged)*100:.1f}%)")
print(f"  Yellow: {ml_yellow} ({ml_yellow/len(merged)*100:.1f}%)")
print(f"  Red: {ml_red} ({ml_red/len(merged)*100:.1f}%)")

print(f"\nActual distribution (after overrides):")
print(f"  Green: {actual_green} ({actual_green/len(merged)*100:.1f}%)")
print(f"  Yellow: {actual_yellow} ({actual_yellow/len(merged)*100:.1f}%)")
print(f"  Red: {actual_red} ({actual_red/len(merged)*100:.1f}%)")

print(f"\nChanges from overrides:")
print(f"  Green: {actual_green - ml_green:+d}")
print(f"  Yellow: {actual_yellow - ml_yellow:+d}")
print(f"  Red: {actual_red - ml_red:+d}")

print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)

if ml_red > 150:
    print("\n🔴 PROBLEM: ML MODEL IS PREDICTING TOO MANY HIGH SCORES")
    print("-" * 80)
    print(f"\nThe ML model assigns scores >= 0.80 to {ml_red} patients ({ml_red/len(merged)*100:.1f}%)")
    print("\nThis means during training, the synthetic labels were TOO AGGRESSIVE.")
    print("Too many training examples were labeled 'high risk', so the model")
    print("learned to predict high risk for many patterns in your data.")
    print("\n🔧 FIX NEEDED: Create V3 with MUCH stricter synthetic label criteria.")
    print("\nSpecifically:")
    print("  - Raise HR threshold from 100 to 105-110")
    print("  - Raise activity threshold to bottom 5% instead of 10%")
    print("  - Lower sleep threshold to < 4.5 hours")
    print("  - Raise sedentary ratio to > 90%")
    
elif abs(actual_red - ml_red) > 50:
    print("\n⚠️  PROBLEM: OVERRIDE RULES ARE CHANGING TOO MANY")
    print("-" * 80)
    overrides_changed = abs(actual_red - ml_red) + abs(actual_yellow - ml_yellow) + abs(actual_green - ml_green)
    print(f"\nOverride rules changed {overrides_changed//2} patient classifications")
    print("\n🔧 FIX NEEDED: Make override rules even more extreme")
    
else:
    print("\n✅ No clear single cause - combination of issues")
    print("-" * 80)
    print("\nBoth ML and overrides contributing to high red %")

conn.close()

print("\n" + "="*80)
print("COMPLETE - Share this output for next steps!")
print("="*80)
