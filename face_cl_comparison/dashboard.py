#!/usr/bin/env python3
"""
Interactive dashboard for viewing experiment results.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import yaml

st.set_page_config(
    page_title="Face CL Results Dashboard",
    page_icon="🧠",
    layout="wide"
)

@st.cache_data
def load_experiment_results(result_dir):
    """Load results from experiment directory."""
    results_df = pd.read_csv(result_dir / 'results.csv')
    
    with open(result_dir / 'experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    summary_path = result_dir / 'summary.txt'
    summary = summary_path.read_text() if summary_path.exists() else ""
    
    return results_df, config, summary


def main():
    st.title("🧠 Face Recognition CL Results Dashboard")
    
    # Sidebar for experiment selection
    st.sidebar.header("Select Experiment")
    
    results_dir = Path('./results')
    if not results_dir.exists():
        st.error("No results directory found!")
        return
    
    # Get all experiment directories
    experiments = sorted([d for d in results_dir.iterdir() if d.is_dir()], 
                        key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not experiments:
        st.error("No experiments found!")
        return
    
    # Select experiment
    exp_names = [exp.name for exp in experiments]
    selected_exp = st.sidebar.selectbox("Experiment", exp_names)
    
    # Load results
    exp_dir = results_dir / selected_exp
    results_df, config, summary = load_experiment_results(exp_dir)
    
    # Display experiment info
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(config['name'])
        st.text(config['description'])
    with col2:
        st.metric("Total Runs", len(results_df))
    with col3:
        best_acc = results_df['average_accuracy'].max()
        st.metric("Best Accuracy", f"{best_acc:.2%}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🎯 Strategy Comparison", 
        "🏗️ Model Comparison",
        "📈 Detailed Metrics",
        "📝 Summary"
    ])
    
    with tab1:
        # Overview plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Average accuracy by configuration
            fig = px.bar(
                results_df.sort_values('average_accuracy', ascending=True),
                x='average_accuracy',
                y='run_name',
                orientation='h',
                title='Average Accuracy by Configuration',
                labels={'average_accuracy': 'Average Accuracy', 'run_name': 'Configuration'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Forgetting comparison
            if 'forgetting' in results_df.columns:
                fig = px.scatter(
                    results_df,
                    x='average_accuracy',
                    y='forgetting',
                    color='strategy',
                    size='training_time',
                    hover_data=['run_name'],
                    title='Accuracy vs Forgetting Trade-off'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Strategy comparison
        st.subheader("Strategy Performance Comparison")
        
        # Group by strategy
        strategy_stats = results_df.groupby('strategy').agg({
            'average_accuracy': ['mean', 'std', 'max'],
            'forgetting': ['mean', 'std'] if 'forgetting' in results_df.columns else [],
            'training_time': ['mean']
        }).round(4)
        
        st.dataframe(strategy_stats, use_container_width=True)
        
        # Box plot by strategy
        fig = px.box(
            results_df,
            x='strategy',
            y='average_accuracy',
            title='Accuracy Distribution by Strategy',
            points='all'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Model comparison
        st.subheader("Model Architecture Comparison")
        
        if 'model_name' in results_df.columns:
            # Group by model
            model_stats = results_df.groupby('model_name').agg({
                'average_accuracy': ['mean', 'std', 'max'],
                'inference_time': ['mean'] if 'inference_time' in results_df.columns else [],
                'model_size_mb': ['first'] if 'model_size_mb' in results_df.columns else []
            }).round(4)
            
            st.dataframe(model_stats, use_container_width=True)
            
            # Efficiency plot
            if 'inference_time' in results_df.columns:
                fig = px.scatter(
                    results_df,
                    x='inference_time',
                    y='average_accuracy',
                    color='model_name',
                    size='model_size_mb' if 'model_size_mb' in results_df.columns else None,
                    title='Model Efficiency: Accuracy vs Inference Time'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Detailed metrics
        st.subheader("Detailed Metrics")
        
        # Experience-wise accuracy
        exp_cols = [col for col in results_df.columns if col.startswith('exp_') and col.endswith('_acc')]
        if exp_cols:
            # Select configuration to visualize
            selected_run = st.selectbox("Select Configuration", results_df['run_name'].unique())
            run_data = results_df[results_df['run_name'] == selected_run].iloc[0]
            
            # Plot experience curve
            exp_accs = [run_data[col] for col in sorted(exp_cols)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(exp_accs))),
                y=exp_accs,
                mode='lines+markers',
                name='Accuracy',
                line=dict(width=3)
            ))
            fig.update_layout(
                title=f'Experience-wise Accuracy: {selected_run}',
                xaxis_title='Experience',
                yaxis_title='Accuracy',
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show raw data
        with st.expander("View Raw Data"):
            st.dataframe(results_df, use_container_width=True)
    
    with tab5:
        # Summary
        st.subheader("Experiment Summary")
        st.text(summary)
        
        # Best configurations
        st.subheader("Top 5 Configurations")
        top_5 = results_df.nlargest(5, 'average_accuracy')[
            ['run_name', 'strategy', 'model_name', 'average_accuracy', 'forgetting']
        ]
        st.dataframe(top_5, use_container_width=True)
    
    # Download options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Download Results")
    
    # CSV download
    csv = results_df.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{selected_exp}_results.csv",
        mime="text/csv"
    )
    
    # Config download
    config_yaml = yaml.dump(config, default_flow_style=False)
    st.sidebar.download_button(
        label="📄 Download Config",
        data=config_yaml,
        file_name=f"{selected_exp}_config.yaml",
        mime="text/yaml"
    )


if __name__ == '__main__':
    main()