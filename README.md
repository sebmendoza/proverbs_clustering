# Semantic Clustering of the Book of Proverbs
## A Weekend Machine Learning Project

---

## Project Overview

**[Write introduction about the goal: exploring thematic structure of Proverbs using semantic embeddings and clustering algorithms]**

---

## Initial Data Exploration

**[Write paragraph about the data source and preparation]**

### Dataset Statistics
- **Total verses analyzed:** 915 verses from Proverbs chapters 1-31
- **Data format:** JSON structure with nested chapters and verses
- **Text source:** English Standard Version (ESV) translation
- **Data cleaning:** Applied via `scripts/cleanup_proverbs.py`
  - Removed reference markers (e.g., `[a]`, `[b]`)
  - Normalized whitespace and semicolons
  - Preserved verse integrity (no splitting or merging)

**[Mention any interesting patterns you noticed in the raw data, like verse length distribution, common themes, etc.]**

---

## Data Cleaning and Embedding Generation

### Data Cleaning Philosophy

**[Explain your decision to keep duplicate verses]**
- **Duplicate verses:** Intentionally left in the dataset
- **Reasoning:** If clustering algorithm works correctly, duplicate/similar verses should cluster together naturally
- **This serves as a validation mechanism** for the clustering approach

**[Explain why you didn't remove punctuation]**
- **Punctuation preservation:** All punctuation kept intact
- **Reasoning:** Using transformer-based language model (`sentence-transformers/all-MiniLM-L6-v2`) that was pre-trained on natural text with punctuation
- **Transformers are context-aware:** They understand punctuation as part of meaning (e.g., questions, emphasis)

### Embedding Generation

**[Describe the embedding process and technical details]**

#### Technical Specifications
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
  - 384-dimensional embeddings
  - Trained on semantic similarity tasks
  - Optimized for sentence-level representations
- **Caching:** Embeddings saved to `esv_embeddings.pkl` for reusability
- **Implementation:** Located in `embeddings.py`
- **Processing:** Each verse encoded independently to capture its semantic meaning

**[Explain why you chose this particular embedding model]**

---

## Clustering Approaches

### 1. Community Graph Clustering with Leiden Algorithm

**[Write introduction explaining what community detection is and why you tried it]**

#### Methodology
- **Similarity graph construction:**
  - Computed cosine similarity matrix between all verse embeddings
  - Created edges between verses with similarity > threshold (0.48)
  - Explored multiple thresholds: 0.35, 0.40, 0.44, 0.46, 0.48, 0.50, etc.
  
- **Threshold selection process:**
  - Analyzed trade-offs: too low = dense graph (hard to find communities), too high = sparse graph (many isolated nodes)
  - Evaluated connected components at each threshold
  - Chose 0.48 as balance between connectivity and meaningful separation

- **Leiden algorithm:**
  - State-of-the-art community detection (improvement over Louvain)
  - Optimizes modularity: measures density within communities vs between communities
  - Applied to largest connected component of graph
  - Implementation: `community_graph/community_graph.py`

#### Visualizations Created
- **2D Force-Directed Layout:**
  - Barnes-Hut physics simulation
  - Nodes colored by community
  - Interactive HTML output (`proverbs_before_leiden.html`, `proverbs_after_leiden.html`)
  
- **3D PCA Projection:**
  - Reduced 384-dimensional embeddings to 3D using Principal Component Analysis
  - Preserved maximum variance for visualization
  - Interactive Plotly visualizations

**[Discuss what insights you gained from the community graphs, or any limitations you observed]**

---

### 2. Hierarchical Clustering

**[Write about why hierarchical clustering is useful for this data]**

#### Why Hierarchical Clustering?
**[Explain the key advantage: ability to see structure at multiple levels of granularity]**
- Shows thematic relationships at different scales (broad themes → specific topics)
- No need to pre-specify number of clusters
- Creates dendrogram showing merge history

#### Implementation Details

**Technical Specifications:**
- **Linkage methods tested:** 
  - `average` (UPGMA): Mean distance between all point pairs (most common)
  - `complete`: Maximum distance (compact clusters)
  - `single`: Minimum distance (elongated clusters, chaining problem)
  - `ward`: Minimizes within-cluster variance (only with Euclidean distance)
  
- **Distance metrics tested:**
  - `cosine`: Angle between embedding vectors (semantic similarity)
  - `euclidean`: Straight-line distance in embedding space
  
- **Best combination:** `average` linkage with `cosine` distance
  - **Cophenetic correlation:** [Add your actual value here - this measures how well dendrogram preserves pairwise distances]

#### Dendrograms Generated

**[Write about the three types of dendrograms you created]**

- **Truncated dendrogram** (`dendrogram.png`):
  - Shows last 50 merges for readability
  - Provides high-level thematic overview
  
- **Full dendrogram** (`dendrogram_full.png`):
  - All 915 verses displayed
  - Shows complete hierarchical structure
  
- **Interactive dendrogram** (`dendrogram_interactive.html`):
  - Plotly visualization with zoom/pan capabilities
  - Hover shows verse chapter:verse and text preview
  - Each leaf node colored uniquely using HSL color space

#### Color Meanings in Dendrograms

**[Explain what the colors represent]**
- **Branch colors:** Different colors indicate distinct communities/clusters formed at that merge height
- **Color threshold:** Horizontal line where coloring changes (set at median merge height)
  - Branches merged **below** threshold get distinct colors (major clusters)
  - Branches merged **above** threshold colored gray (higher-level groupings)
- **Merge height (Y-axis):** Distance at which clusters merge
  - Lower values = very similar verses merged early
  - Higher values = dissimilar groups merged late

**[Discuss what patterns you noticed in the dendrogram - did certain themes cluster together?]**

---

### 3. K-Means Clustering

**[Write introduction about K-means and why it's a standard baseline for clustering]**

#### Experimental Process

**Range of k values tested:**
- Initial broad sweep: k = 2 to 29
- Focused analysis: k = 10 to 40
- **Final chosen value: k = 20**

#### Quality Metrics Computed

**[Explain each metric and why it matters]**

1. **Silhouette Score** (range: -1 to 1, higher is better)
   - Measures how similar a point is to its own cluster vs. other clusters
   - **Your dataset's scores:** [Mention the range you observed - likely 0.15-0.30]
   - **Why generally low for Proverbs:**
     - Short text (average ~15-20 words per verse)
     - Overlapping themes across clusters
     - High semantic similarity between many verses
     - This is expected and normal for short texts with shared vocabulary

2. **Davies-Bouldin Score** (lower is better)
   - Ratio of within-cluster to between-cluster distances
   - Complements silhouette score

3. **Inertia** (lower is better, but always decreases with more clusters)
   - Sum of squared distances to cluster centers
   - Used to identify "elbow" in curve

4. **Cluster Size Distribution**
   - Checked for imbalanced clusters (size ratio > 10 = warning)
   - Flagged tiny clusters (< 3 verses = potential noise)
   - Monitored small clusters (< 5 verses)

#### Why K=20 Was Chosen

**[Explain your decision-making process]**

1. **Silhouette score analysis:** [Describe where k=20 fell on the curve]
2. **Round number preference:** Easier to interpret and present
3. **Manual cluster inspection:**
   - Verified each cluster had coherent theme
   - Ensured no clusters were purely noisy
   - Checked that themes weren't overly generic (too few clusters) or overly specific (too many clusters)
4. **Balance between:** Granularity of themes vs. interpretability

---

### 4. UMAP Visualization

**[Write about using UMAP for dimensionality reduction]**

#### What is UMAP?
**[Explain: Uniform Manifold Approximation and Projection - superior to PCA and t-SNE for preserving both local and global structure]**

#### Implementation

**Exploration conducted:**
- **2D projections:** Grid search over parameters
  - `n_neighbors`: [5, 15, 30, 50] - controls local vs global structure
  - `metric`: ['euclidean', 'cosine', 'manhattan']
  - `min_dist`: [0.0, 0.1, 0.25, 0.5] - minimum distance between points in projection
  
- **3D projections:** Interactive Plotly visualizations
  - Color-coded by cluster assignment
  - Allows rotation and inspection of cluster separation

**Parameter files:**
- `umapviz.py` contains all UMAP exploration functions
- Generates systematic grids comparing parameter combinations

#### Results

**[Describe what you observed]**
- **Some separation visible:** Clusters showed visual grouping but with overlap
- **Not perfect separation:** Expected due to:
  - Semantic overlap between themes in Proverbs
  - High dimensionality of original embeddings (384D → 2D/3D = information loss)
  - Short text nature of individual verses
  
**[Mention which parameter combination worked best for your data]**

---

## Cluster Title Generation Experiments

**[Write introduction about the challenge of automatically labeling clusters]**

### Approach 1: Most Frequent Words

**[Describe this simple baseline approach]**

#### Implementation
- Script: `scripts/use_freq_for_titles.py`
- Method: Extract most common words per cluster (excluding stop words)
- Example outputs (from `outputs/cluster_frequency.txt`):
  - Cluster 0: "sluggard" (appears 11 times)
  - Cluster 1: "wisdom" (12), "knowledge" (8)
  - Cluster 12: "wicked" (53), "righteous" (37)
  - Cluster 19: "lord" (41), "fear" (12)

#### Limitations
**[Explain why this didn't work well enough]**
- Often too generic ("the", "of", "his")
- Single words lack context
- Doesn't capture thematic essence
- No grammatical structure

---

### Approach 2: KeyBERT Keyword Extraction

**[Describe KeyBERT and why you thought it would work]**

#### Implementation
- Script: `keyword_extraction.py`
- Method: BERT-based keyword extraction with semantic similarity
- Attempted to extract n-grams (1-5 words) that best represent cluster themes

#### Results
**[Explain why this also performed poorly]**
- Extracted phrases often too literal
- Didn't generalize well across clusters
- Still lacked coherent thematic summary
- Better than frequency but insufficient for quality titles

---

### Approach 3: Local LLM with Ollama (Llama 3)

**[Describe your attempt with local LLMs]**

#### Setup
- Tool: Ollama running Llama 3 locally
- Model size: 4.7GB
- Script: `kmeans/cluster_titles.py` with OllamaBackend
- Documentation: `kmeans/OLLAMA_SETUP.md`

#### Prompting Strategy
```
You are analyzing Bible verses from the Book of Proverbs. 
Your task is to create a concise thematic title for this cluster of verses.

Verses:
[List of verses]

Task: Generate a {word_count}-word title that captures the main theme.
```

#### Challenges
**[Describe the problems you encountered]**
- **Generation speed:** Very slow (3-8 seconds per cluster title)
- **For k=20 with 3 title lengths:** 20 clusters × 3 titles = 60 LLM calls = several minutes
- **Quality:** Better than keywords but not amazing
- **Consistency:** Hard to enforce exact word counts
- **Resource usage:** Required keeping Ollama service running

**[Mention if the titles were good enough, or still needed improvement]**

---

### Approach 4: Claude Sonnet 4.5 (Batch API)

**[Write about your final successful approach]**

#### Why Claude?
**[Explain your reasoning for switching to Claude]**
- Superior language understanding
- Better at following specific constraints (word count)
- Batch API for efficient processing
- High-quality theological/thematic understanding

#### Implementation Details

**Technical setup:**
- Script: `scripts/use_claude_for_titles.py`
- Model: `claude-sonnet-4-5-20250929`
- **System prompt:** "You are a Biblical Scholar and Theologian. You provide 4 word responses only."
- **Max tokens:** 50 (enforcing brevity)
- **Batch processing:** All 20 cluster requests submitted as single batch

**Prompt template:**
```
Here are verses from the Book of Proverbs that were grouped together 
by semantic similarity. Provide a 4-word thematic summary that captures 
what these verses have in common. Just the 4 words, nothing else.

[Verses]
```

#### Results

**[Write about the quality of Claude-generated titles]**
- Much more coherent and thematic
- Better theological understanding
- Consistent formatting
- Appropriate level of abstraction
- Actually captured the essence of clusters

**Examples:** [Add some of your best cluster titles from Claude here]

---

## K=20 Clusters: Titles vs. Frequent Words

**[Write a comparative analysis section]**

**Create a table or list showing:**
- Cluster ID
- Claude's 4-word title
- Top frequent words from that cluster
- Your observation about how well they match

**Example structure:**
| Cluster | Claude Title | Top Frequent Words | Analysis |
|---------|--------------|-------------------|----------|
| 0 | [Title] | sluggard (11) | [How well do they align?] |
| 1 | [Title] | wisdom (12), knowledge (8) | [Commentary] |
| 12 | [Title] | wicked (53), righteous (37) | [Commentary] |

**[Discuss the patterns: Do Claude's titles generally align with frequent words? When do they diverge and why?]**

---

## Project Structure and Code Organization

**[Brief write-up about how the code is organized]**

### Main Modules
- `embeddings.py` - Embedding generation and caching
- `utils.py` - Data loading and common utilities
- `main.py` - Entry point for running analyses

### Clustering Implementations
- `kmeans/` - K-means clustering pipeline
  - `kmeans.py` - Core K-means implementation
  - `clustering_pipeline.py` - Full automated pipeline
  - `cluster_quality.py` - Quality metrics computation
  - `cluster_viz.py` - 2D/3D visualizations
  - `cluster_titles.py` - Multi-backend title generation (Ollama/Transformers)
  - `cluster_analysis.py` - Results analysis and comparison
  
- `hierarchical/` - Hierarchical clustering
  - `hierarchical_clustering.py` - Dendrogram generation
  
- `community_graph/` - Graph-based clustering
  - `community_graph.py` - Leiden algorithm implementation
  
- `dcscan/` - DBSCAN experiments (intentionally excluded from this report)
  - Poor results, minimal time investment

### Scripts
- `scripts/cleanup_proverbs.py` - Data preprocessing
- `scripts/use_claude_for_titles.py` - Claude Batch API integration
- `scripts/use_freq_for_titles.py` - Frequency-based titles
- `scripts/regenerate_titles.py` - Batch title regeneration

### Outputs
- `outputs/esv_kmeans_19_clusters.json` - Final clustering results (k=20, but filename says 19 - note this)
- `outputs/cluster_frequency.txt` - Frequency analysis
- `visualizations/` - Generated plots and interactive visualizations

---

## Key Findings and Insights

**[Write a reflective section about what you learned]**

### Technical Insights

1. **Embeddings quality matters more than clustering algorithm**
   - [Your thoughts]

2. **Short text clustering is inherently challenging**
   - Low silhouette scores are expected, not a failure
   - [More thoughts]

3. **Domain-specific LLMs (Claude) significantly outperform generic approaches**
   - [Your observations]

4. **Hierarchical methods provide interpretability benefits**
   - [Your insights]

### Thematic Discoveries

**[What did you learn about Proverbs from this analysis?]**
- Which themes naturally cluster together?
- Any surprising groupings?
- Do the clusters align with traditional scholarly divisions of Proverbs?

---

## Limitations and Future Work

### Current Limitations

1. **Weekend project scope:**
   - [Limited time for hyperparameter tuning]
   - [Didn't explore all possible clustering methods]
   - [...]

2. **Silhouette scores generally low (0.15-0.30):**
   - **This is expected for short text documents**
   - Verses share vocabulary and themes
   - Not a failure, but reflects the nature of the data

3. **Duplicate verses:**
   - Left in intentionally as validation
   - Could be deduplicated in future analysis

4. **Single translation (ESV):**
   - Different translations might yield different clusterings
   - Semantic nuances could shift groupings

5. **K-means assumptions:**
   - Assumes spherical clusters
   - Sensitive to initialization (mitigated with random_state)
   - Requires pre-specifying k

### Future Directions

**[List potential improvements and extensions]**

1. **Multi-translation analysis:**
   - Compare clustering across ESV, NIV, KJV, NASB
   - Identify translation-invariant themes

2. **Hierarchical labels at multiple levels:**
   - Generate titles at different cut heights
   - Create taxonomy of themes

3. **Topic modeling approaches:**
   - LDA (Latent Dirichlet Allocation)
   - BERTopic for transformer-based topic modeling
   - Compare with clustering results

4. **Cross-book analysis:**
   - Extend to other wisdom literature (Ecclesiastes, Job, Psalms)
   - Find inter-book thematic connections

5. **Interactive exploration tool:**
   - Web interface for browsing clusters
   - Search by theme or verse
   - Visualize connections

6. **Fine-tuned embeddings:**
   - Train embeddings specifically on Biblical text
   - Might capture theological nuances better

---

## Technologies Used

### Core Libraries
- **Python 3.x**
- **scikit-learn** - K-means, metrics, hierarchical clustering
- **sentence-transformers** - Embedding generation
- **numpy** - Array operations
- **scipy** - Hierarchical clustering, distance metrics

### Visualization
- **matplotlib** - Static plots
- **plotly** - Interactive 3D visualizations
- **pyvis** - Network graph visualizations

### Graph Analysis
- **networkx** - Graph construction and manipulation
- **python-igraph** - Graph algorithms
- **leidenalg** - Community detection

### Dimensionality Reduction
- **umap-learn** - UMAP projections
- **scikit-learn PCA** - Principal Component Analysis

### NLP & LLMs
- **KeyBERT** - Keyword extraction
- **transformers** (HuggingFace) - Fallback LLM
- **anthropic** - Claude API
- **ollama** (via requests) - Local LLM hosting

### Utilities
- **json** - Data storage
- **pickle** - Embedding caching
- **pathlib** - File operations
- **logging** - Debug output
- **python-dotenv** - Environment management

---

## How to Run This Project

### Prerequisites
```bash
pip install -r requirements.txt
```

### Generate Embeddings
```bash
python embeddings.py
```

### Run Clustering Experiments

**K-means with full pipeline:**
```bash
python main.py
```

**Hierarchical clustering:**
```python
python -c "from main import run_hierarchical_pipeline; run_hierarchical_pipeline(...)"
```

### Generate Cluster Titles

**With Claude (requires API key):**
```bash
export ANTHROPIC_API_KEY="your-key-here"
python scripts/use_claude_for_titles.py
```

**With Ollama (requires Ollama installed):**
```bash
ollama serve  # In one terminal
python main.py  # In another terminal
```

See `kmeans/CLUSTERING_PIPELINE_README.md` for detailed usage instructions.

---

## Conclusion

**[Write a concluding paragraph summarizing:]**
- What you set out to do
- What you accomplished
- Key takeaways
- What you learned about ML, NLP, or Proverbs
- Why this was a valuable learning experience

---

## Acknowledgments

**[Optional: Mention any resources, papers, or people that helped]**
- Sentence-Transformers library
- Leiden algorithm paper
- Any tutorials or guides you followed

---

## License and Data

- **Code:** [Add your preferred license - MIT, GPL, etc.]
- **Biblical Text:** English Standard Version (ESV), copyright Crossway
- **Purpose:** Educational and research purposes only

---

**Project Timeline:** Weekend project, [Add dates]  
**Author:** Sebastian Mendoza  
**Repository:** [Add GitHub link if applicable]
