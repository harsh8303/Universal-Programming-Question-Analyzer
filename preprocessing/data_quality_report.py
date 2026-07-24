# preprocessing/data_quality_report.py
import pandas as pd
import os
import sys

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config

def generate_report():
    print(" Starting Phase 3: Generating Data Quality Report...")
    
    # Path to cleaned dataset
    input_path = os.path.join(config.DATA_DIR, "clean", "cleaned_programming_problems.csv")
    
    # Path for reports directory
    reports_dir = os.path.join(config.DATA_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    output_path = os.path.join(reports_dir, "data_quality_report.csv")
    
    if not os.path.exists(input_path):
        print(f" ERROR: Cleaned dataset not found at {input_path}")
        sys.exit(1)
        
    print("Loading Cleaned Dataset...")
    df = pd.read_csv(input_path)
    
    print(" Calculating Platform-wise Metrics...")
    
    # Calculate metrics platform-wise
    platforms = df['platform'].unique()
    report_data = []
    
    for platform in platforms:
        pdf = df[df['platform'] == platform]
        
        total_problems = len(pdf)
        valid_desc = pdf['has_description'].sum()
        missing_desc = total_problems - valid_desc
        missing_tags = (pdf['num_tags'] == 0).sum()
        avg_desc_len = round(pdf['description_length'].mean(), 2) if total_problems > 0 else 0
        avg_word_count = round(pdf['word_count'].mean(), 2) if total_problems > 0 else 0
        
        report_data.append({
            'Platform': platform,
            'Total Problems': total_problems,
            'Valid Descriptions': valid_desc,
            'Missing Descriptions': missing_desc,
            'Missing Tags': missing_tags,
            'Avg Description Length': avg_desc_len,
            'Avg Word Count': avg_word_count
        })
        
    # Create DataFrame for the report
    report_df = pd.DataFrame(report_data)
    
    # Add an OVERALL row at the bottom
    overall_data = {
        'Platform': 'OVERALL',
        'Total Problems': len(df),
        'Valid Descriptions': df['has_description'].sum(),
        'Missing Descriptions': (~df['has_description']).sum(),
        'Missing Tags': (df['num_tags'] == 0).sum(),
        'Avg Description Length': round(df['description_length'].mean(), 2),
        'Avg Word Count': round(df['word_count'].mean(), 2)
    }
    
    # Append overall data using pandas concat (modern way)
    report_df = pd.concat([report_df, pd.DataFrame([overall_data])], ignore_index=True)
    
    # Save the report
    print(" Saving report to CSV...")
    report_df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print("SUCCESS: PHASE 3 DATA QUALITY REPORT GENERATED! ")
    print("="*70)
    # Print the DataFrame nicely in the terminal
    print(report_df.to_string(index=False))
    print("="*70)
    print(f" Report saved at: {output_path}")
    print(" Ready for Phase 4: Tokenizer (tokenizer.py)!")

if __name__ == "__main__":
    generate_report()