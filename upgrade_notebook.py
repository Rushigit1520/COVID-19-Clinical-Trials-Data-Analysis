import nbformat
import re

def upgrade_notebook(input_path, output_path):
    # Read the notebook
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    imports_found = False

    for cell in nb.cells:
        if cell.cell_type == 'code':
            source = cell.source
            original_source = source

            # Replace the CSV path
            if 'unified mentor projects/COVID clinical trials (2).csv' in source:
                source = source.replace('unified mentor projects/COVID clinical trials (2).csv', 'COVID clinical trials (2).csv')
            
            # Inject Seaborn Theme configuration after imports
            if 'import seaborn as sns' in source and not imports_found:
                imports_found = True
                # Add sns theme setup
                if 'sns.set_theme' not in source:
                    source += '\n\nsns.set_theme(style="whitegrid")'

            # Fix layout for all matplotlib plots
            if 'plt.show()' in source:
                # Add tight_layout before show if not already there
                if 'plt.tight_layout()' not in source:
                    source = source.replace('plt.show()', 'plt.tight_layout()\nplt.show()')

            # We can also add interactive plotly for some cells
            # E.g., the Status count
            if "sns.barplot(x=status_counts.index, y=status_counts.values" in source:
                source = '''fig = px.bar(x=status_counts.index, y=status_counts.values, 
             labels={'x': 'Status', 'y': 'Number of Studies'},
             title="Number of COVID-19 Clinical Trials by Status",
             color=status_counts.index, color_discrete_sequence=px.colors.qualitative.Pastel)
fig.show()'''
            
            # Phases count
            if "sns.barplot(x=phase_counts.index, y=phase_counts.values" in source:
                source = '''fig = px.pie(values=phase_counts.values, names=phase_counts.index, 
             title="Distribution of COVID-19 Clinical Trials by Phase", hole=0.3,
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()'''

            # WordCloud for Conditions
            if "sns.barplot(x=condition_counts.index, y=condition_counts.values" in source:
                if 'wordcloud' not in source.lower():
                    # Keep the bar plot
                    source = source.replace('plt.show()', "plt.tight_layout()\nplt.show()")
                    
                    # Append a WordCloud
                    source += '''\n\n# Also display a Word Cloud for a better aesthetic
text = " ".join(df['Conditions'].dropna().astype(str).tolist())
if len(text) > 0:
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Clinical Trial Conditions", fontsize=16)
    plt.tight_layout()
    plt.show()'''

            cell.source = source

    # Write the modified notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    upgrade_notebook('unified covid prj_Original.ipynb', 'unified covid prj.ipynb')
    print("Notebook upgraded successfully!")
