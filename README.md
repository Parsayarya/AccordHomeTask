# **One must imagine Sisyphus happy** 

I leave Sisyphus at the foot of the mountain. One always finds one's burden again. But Sisyphus teaches the higher fidelity that negates the gods and raises rocks. He too concludes that all is well. This universe henceforth without a master seems to him neither sterile nor futile. Each atom of that stone, each mineral flake of that night-filled mountain, in itself, forms a world. The struggle itself toward the heights is enough to fill a man's heart. One must imagine Sisyphus happy.


-Albert Camus

**So, simply put, I loved doing the challenge :)))**


# Accord Message Topic Classification 


## Table of Contents

- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)
- [EDA details](#eda-details)
- [Future Improvements](#future-improvements)


## Methodology

### Exploratory Data Analysis (EDA)
**Module:** `utils/EDA.py`

- Topic distribution across messages
- Most common topic co-occurrences (pairs)
- Topics frequently appearing in ambiguous contexts
- Confidence score distributions
- Global vs. custom topic usage patterns

**Findings:**
- ~42% of messages have only 1 topic
- ~30% have 2 topics (highest ambiguity potential)
- "Gameplay" appears in 10,794 ambiguous multi-topic messages
- Mean confidence: 0.78, Median: 0.70
- Significant tail of low-confidence classifications

### PMI-Based Contextual Polysemy Detection
**Module:** `utils/PMI-CP-Detector.py`

Detects messages with vocabulary atypical for their assigned topic using **Pointwise Mutual Information (PMI)**:

```
PMI(token, topic) = log₂(P(token|topic) / (P(token) × P(topic)))
```

1. Build token-topic co-occurrence statistics per community
2. Calculate PMI scores for each token-topic pair
3. Score messages based on average PMI of their tokens
4. Flag outliers with Z scores (low PMI = contextually unusual vocabulary)

**Output:** Community-specific CSVs (here only one) identifying messages likely misclassified due to contextual ambiguity.

### Composite Ambiguity Ranking
**Module:** `src/ranker.py`

This here I pretty much made up but looking at the result with some validation could be a nice metric to track.


**Features:**
- **Entropy** (0.50 weight): Uncertainty across topic assignments
- **Inverse Margin** (0.35 weight): 1 - (top1_conf - top2_conf)
- **Topic Count** (0.15 weight): Number of assigned topics

```python
ambiguity_score = 0.50 × entropy + 0.35 × inv_margin + 0.15 × (topic_count - 1)/4
```

**Flagging Logic:**
- Messages with multiple topics AND score ≥ 90th percentile in their community

**Output:** `outputs/top_ambiguous_ranked.csv` with ranked ambiguous messages

### Semantic Retrieval + Gemini Re-classification
**Module:** `src/Classify.py`

Improves classification through context-aware re-classification:


Thanks to cluade, the beatiful architecture demo below :))))


**Architecture:**
```
Multi-topic Message
        ↓
   [Embedding]
        ↓
   Retrieve K=10 nearest single-topic, high-confidence examples
        ↓
   Build Prompt:
     - Community context
     - Topic definitions
     - Similar message examples
     - Target message
        ↓
   Gemini API (gemini-2.5-flash)
        ↓
   Single topic prediction + confidence
```

### Performance Metrics
**Module:** `utils/metrics.py`

evaluation metrics (requires manual labeling):

**Metrics:**
- **Accuracy**: 74% (95% CI: 64.6% - 81.6%)
- **Error Rate**: 26%
- **Brier Score**: 0.223 (calibration quality)
- **AUC**: 0.669 (confidence discriminates correctness)
- **Average Precision**: 0.819
- **ECE (10 bins)**: 0.197 (expected calibration error)

**Confidence Statistics:**
- Mean confidence (overall): 0.937
- Mean confidence (correct): 0.943
- Mean confidence (incorrect): 0.920

## Project Structure

```
AccordHomeTask/
├── main.py                          # Main pipeline orchestrator
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── data/                            # Input data (CSV files from PostgreSQL)
│   ├── messages.csv
│   ├── message_topic_classifications.csv
│   ├── topic.csv
│   └── community.csv
│
├── utils/                           # Utility modules
│   ├── export_PostGres.py           # PostgreSQL export utility
│   ├── EDA.py                       # Exploratory data analysis
│   ├── PMI-CP-Detector.py           # PMI-based ambiguity detection
│   └── metrics.py                   # Performance metrics calculation
│
├── src/                             # Core classification modules
│   ├── Classify.py                  # Gemini re-classification pipeline
│   └── ranker.py                    # Ambiguity ranking system
│
└── outputs/                         # Generated outputs
    ├── eda/                         # EDA visualizations
    │   ├── topic_distribution.png
    │   ├── top_topic_pairs.png
    │   ├── ambiguous_topics.png
    │   ├── confidence_score_distribution.png
    │   └── topic_usage_statistics.png
    │
    ├── performance/                 # Performance metrics & plots
    │   └── [metrics visualizations]
    │
    ├── top_ambiguous_ranked.csv     # Ranked ambiguous messages
    └── gemini_classification_candidates.csv  # Re-classification results
```

## Setup & Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (I ran it on my own PC, it's not that bad)
- PostgreSQL database 
- Google Gemini API key

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Parsayarya/AccordHomeTask.git
cd AccordHomeTask
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
# Required for Gemini classification
export GEMINI_API_KEY='your-gemini-api-key-here'

# Optional: for PostgreSQL export
export ACCORD_PG_DSN='postgresql://user:password@host:port/database'
```

### Data

```bash
python utils/export_PostGres.py --dsn "$ACCORD_PG_DSN" --out data/
```

## Usage

### Run Full Pipeline

```bash
# Run complete pipeline (assumes data/ exists)
python main.py --skip-export
```

### Execution

**Export Data from PostgreSQL:**
```bash
python main.py --dsn "postgresql://..." --export-dir data
```

**Run Exploratory Data Analysis:**
```bash
python main.py --skip-export --skip-pmi --skip-ranker --skip-classification
```

**Detect Ambiguous Messages (PMI):**
```bash
python utils/PMI-CP-Detector.py
```

**Rank Messages by Ambiguity:**
```bash
python main.py --skip-export --skip-eda --skip-pmi --skip-classification
```

**Run Re-classification:**
```bash
# Ensure GEMINI_API_KEY is set
python main.py --skip-export --skip-eda --skip-pmi --skip-ranker --sample-n 100
```

**Calculate Metrics (after manual labeling):**
```bash
# First: manually add 'label' column to outputs/gemini_classification_candidates.csv
# Then:
python main.py --metrics-only
```

### Command-Line Options

```bash
python main.py [OPTIONS]

Options:
  --dsn TEXT              PostgreSQL connection string
  --export-dir TEXT       Directory for exported data (default: data)
  --skip-export           Skip PostgreSQL export step
  --skip-eda              Skip EDA step
  --skip-pmi              Skip PMI detection step
  --skip-ranker           Skip ambiguity ranking step
  --skip-classification   Skip Gemini classification step
  --metrics-only          Only calculate metrics (requires labeled data)
  --data-dir TEXT         Directory containing input data (default: data)
  --eda-output TEXT       Directory for EDA outputs (default: outputs/eda)
  --sample-n INT          Number of messages to classify (default: 100)
  --chunk-size INT        Chunk size for PostgreSQL export (default: 100000)
```

## Results

### Classification Performance (100 manually labeled samples)

| Metric | Value | my interpretations |
|--------|-------|----------------|
| **Accuracy** | 74.0% | Correct classifications |
| **95% Confidence Interval** | (64.6%, 81.6%) | Statistical reliability |
| **Error Rate** | 26.0% | Incorrect classifications |
| **Brier Score** | 0.223 | Calibration quality (lower is better) |
| **AUC** | 0.669 | Confidence discriminates correctness |
| **Average Precision** | 0.819 | High precision at top confidence |
| **ECE (10 bins)** | 0.197 | Expected calibration error |

### Confidence Analysis

![Confidence Distribution](outputs/performance/metrics.png)

**Key Observations:**
1. **High Overall Confidence**: Mean = 0.937, showing model certainty
2. **Similar Confidence for Errors**: Mean = 0.920 (errors) vs 0.943 (correct)
   - Suggests need for better uncertainty quantification
3. **Strong Calibration**: AUC = 0.819 indicates confidence correlates with correctness

### Comparison: Baseline vs. new System

| Aspect | Baseline System | new System |
|--------|----------------|-----------------|
| Context Awareness | None |  Semantic retrieval + community context |
| Examples |  None |  10 high-quality similar messages |
| Topic Definitions |  Yes |  Yes (enhanced in prompt) |
| Ambiguity Detection |  None |  PMI + Ranking + Multi-signal analysis |
| Confidence Calibration |  Basic |  Improved (ECE = 0.197) |

## EDA details

### Topic Distribution
![Topic Distribution](outputs/eda/topic_distribution.png)

- **42%** of messages: Single topic (unambiguous)
- **30%** of messages: 2 topics (highest ambiguity risk)
- **Long tail**: Up to 15 topics per message

### Most Ambiguous Topics
![Ambiguous Topics](outputs/eda/ambiguous_topics.png)

**Top 5 Topics in Ambiguous Classifications:**
1. **Gameplay**: 10,794 occurrences (most generic)
2. **Items**: 5,616 occurrences
3. **Ships**: 3,959 occurrences
4. **Community**: 3,845 occurrences
5. **Balance**: 3,723 occurrences

### Topic Co-occurrence Patterns
![Top Topic Pairs](outputs/eda/top_topic_pairs.png)

**Most Common Pairs:**
1. Gameplay + Gameplay: 84,656 (self-pair, indicating multi-faceted gameplay discussions)
2. Items + Gameplay: 77,185
3. Ships + Community: 63,291

### Confidence Score Distribution
![Confidence Distribution](outputs/eda/confidence_score_distribution.png)

**Stats:**
- Mean: 0.775
- Median: 0.700
- **60%** of classifications have confidence ≥ 0.70
- **step distribution?**: we have only three values right?

## Future Improvements

### Continuous Learning Loop
- **Automated Labeling**: Use high-confidence re-classifications to create training data
- **Fine-tuning**: Train domain-specific classifier on gaming terminology

### Topic Definition 
- **Dynamic Definitions**: Update based on community-specific usage patterns
- **Negative Examples**: Add "NOT this" examples to topic definitions

### Integration & Deployment
- **Real-time Pipeline**: Integrate with existing Accord infrastructure
- **Monitoring Dashboard**: Track classification quality metrics
- **Feedback Loop**: Allow community moderators to correct classifications
- **A/B Testing**: Compare old vs. new system performance


---

**Author**: Parsa Yaryab
**Repository**: https://github.com/Parsayarya/AccordHomeTask  
**Date**: 2025
