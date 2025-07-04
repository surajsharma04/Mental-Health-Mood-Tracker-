﻿# Mental-Health-Mood-Tracker-
# ✨ Mindful Metrics - An Intelligent Mood Tracker

Mindful Metrics is a Python-based command-line application designed to help users track their daily mood and gain intelligent insights into their mental well-being. This project goes beyond simple data logging by implementing AI/ML techniques to analyze trends, identify patterns, and recognize concerning deviations in mood data.

The system is built with a modular architecture, prioritizing safety, privacy, and empathetic, actionable feedback.

---

## 🚀 Key Features

*   **Interactive Data Entry:** A user-friendly command-line interface to log daily mood scores (1-10), context tags (e.g., `work`, `exercise`), and journal entries.
*   **Trend Analysis:** Automatically calculates 7-day moving averages and establishes a personal mood baseline to contextualize daily feelings.
*   **AI-Powered Pattern Recognition:** Identifies correlations between activities/tags and mood scores (e.g., "When you 'exercise', your mood is higher than average").
*   **Anomaly Detection:** Statistically flags days or periods where mood is significantly lower than the user's established normal range.
*   **NLP Sentiment Analysis:** Uses VADER to analyze the sentiment of journal entries, providing deeper insights that can contrast with or support numerical mood scores.
*   **Empathetic & Responsible Recommendations:** Based on established patterns and concerning trends (like prolonged low mood), the system provides gentle nudges and, when appropriate, suggests seeking professional help.

---

## 🛠️ Skills Demonstrated

This project showcases a blend of technical and critical thinking skills:

*   **AI/ML:** Implementation of time series analysis, pattern recognition (correlation), anomaly detection, and Natural Language Processing (NLP).
*   **Critical Thinking:** Understanding of mental health indicators, including the logic for when to suggest professional help based on persistent, low-mood trends.
*   **Problem Solving:** Designed to handle subjective data, missing entries, and user privacy concerns.
*   **Modular Structure:** The `MoodAnalyzer` class encapsulates all logic, separating data collection from the analysis pipeline.
*   **Clear Architecture:** A clean pipeline from `User Input` → `Analysis` → `Actionable Insights`.

---

## ⚙️ System Architecture

The application follows a clear, staged pipeline for processing data:

`Raw User Entries` → `[Module 1: Data Ingestion & Preprocessing]` → `[Module 2: Trend & Baseline Analysis]` → `[Module 3: Pattern & Anomaly Detection]` → `[Module 4: Insight Synthesis & Recommendation]` → `User-Facing Report`

---

## 🏁 Getting Started

### Prerequisites

*   Python 3.x
*   Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/mental-health-tracker.git
    ```
    *(Replace `your-username` with your actual GitHub username)*

2.  **Navigate to the project directory:**
    ```bash
    cd mental-health-tracker
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### How to Run the Application

Execute the main script from your terminal:

```bash
python mood_analyzer.py
