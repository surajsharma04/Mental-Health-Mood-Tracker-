
# First, ensure the necessary library for NLP is installed.
# The "-q" makes the installation quiet.
!pip install vaderSentiment -q

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict
from datetime import datetime, timedelta

# The MoodAnalyzer class remains unchanged.
# Its modularity means we don't need to touch it.
class MoodAnalyzer:
    """
    A system that analyzes daily mood entries to identify patterns,
    concerning trends, and provide actionable insights.
    """
    def __init__(self, user_data):
        self.df = self._preprocess_data(user_data)
        self.insights = []
        self.analytics = {}
    def _preprocess_data(self, user_data):
        df = pd.DataFrame(user_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        if 'tags' not in df.columns: df['tags'] = [[] for _ in range(len(df))]
        if 'journal' not in df.columns: df['journal'] = ''
        df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])
        df['journal'] = df['journal'].fillna('')
        return df
    def _analyze_trends_and_baseline(self):
        if len(self.df) < 7: return
        self.analytics['baseline_mood'] = self.df['mood_score'].mean()
        self.analytics['mood_std_dev'] = self.df['mood_score'].std()
        self.df['mood_7_day_avg'] = self.df['mood_score'].rolling(window=7).mean()
    def _detect_patterns_and_anomalies(self):
        if 'baseline_mood' not in self.analytics: return
        anomaly_threshold = self.analytics['baseline_mood'] - (1.75 * self.analytics['mood_std_dev'])
        self.df['is_anomaly'] = self.df['mood_score'] < anomaly_threshold
        tag_moods = defaultdict(lambda: {'scores': [], 'mean': 0})
        exploded_df = self.df.explode('tags').dropna(subset=['tags'])
        for tag, group in exploded_df.groupby('tags'):
            tag_moods[tag]['mean'] = group['mood_score'].mean()
        self.analytics['tag_correlations'] = tag_moods
        analyzer = SentimentIntensityAnalyzer()
        self.df['journal_sentiment'] = self.df['journal'].apply(
            lambda text: analyzer.polarity_scores(text)['compound'] if isinstance(text, str) and text else 0)
    def _synthesize_insights(self):
        if 'baseline_mood' in self.analytics:
            self.insights.append({"type": "info", "text": f"Over the last {len(self.df)} days, your average mood has been {self.analytics['baseline_mood']:.1f}/10."})
        else:
            self.insights.append({"type": "info", "text": "Keep logging your mood! Once you have 7 days of data, we can start showing you trends."})
            return
        baseline = self.analytics['baseline_mood']
        for tag, data in self.analytics['tag_correlations'].items():
            if data['mean'] > baseline + 1.0:
                self.insights.append({"type": "positive_pattern", "text": f"Positive Pattern: When you log '{tag}', your mood is {data['mean']:.1f}/10 on average, which is higher than usual."})
            elif data['mean'] < baseline - 1.0:
                self.insights.append({"type": "negative_pattern", "text": f"Challenging Pattern: Days you log '{tag}' seem to be tougher, with an average mood of {data['mean']:.1f}/10."})
        if self.df['is_anomaly'].tail(14).sum() >= 3:
             self.insights.append({"type": "concerning_trend", "text": "Concerning Trend: We've noticed a few days where your mood was significantly lower than your typical range."})
        if len(self.df) > 14 and self.df['mood_7_day_avg'].iloc[-1] < self.analytics['baseline_mood'] - self.analytics['mood_std_dev']:
            self.insights.append({"type": "care_recommendation", "text": "Care Recommendation: Your average mood has been trending low for a significant period. Considering a chat with a professional is a proactive step towards well-being."})
        conflicting_entry = self.df[(self.df['mood_score'] > 7) & (self.df['journal_sentiment'] < -0.5)]
        if not conflicting_entry.empty:
            date_str = conflicting_entry.index[0].strftime('%B %d')
            self.insights.append({"type": "nlp_insight", "text": f"On {date_str}, you reported a high mood, but your journal entry had a strongly negative sentiment."})
    def run_analysis(self):
        self._analyze_trends_and_baseline()
        self._detect_patterns_and_anomalies()
        self._synthesize_insights()
        return self.insights


# --- UPDATED Interactive Data Collection ---
def collect_user_entries():
    """Interactively collects daily mood entries from the user."""
    # --- CHANGE 1: Improved welcome message ---
    print("--- Welcome to Mindful Metrics ---")
    print("Let's log your mood for the past few days.")
    print("When you're finished entering data, just type 'done'.\n")
    
    user_entries = []
    current_date = datetime.now() 
    
    while True:
        # --- CHANGE 2: The improved input prompt ---
        mood_input = input(f"Enter mood score for {current_date.strftime('%Y-%m-%d')} [1-10] (or type 'done' to finish): ")
        
        if mood_input.lower() == 'done':
            if not user_entries:
                print("No data entered. Exiting.")
                return None
            # --- CHANGE 3: Clearer confirmation message ---
            print("\nInput complete. Generating your report...")
            break
            
        try:
            mood_score = int(mood_input)
            if not 1 <= mood_score <= 10:
                print("Invalid score. Please enter a number between 1 and 10.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number or 'done'.")
            continue

        tag_input = input("Enter any relevant tags, separated by commas (e.g., work, exercise): ")
        tags = [tag.strip() for tag in tag_input.split(',') if tag.strip()]
        journal_input = input("Enter a journal entry (optional, press Enter to skip): ")

        user_entries.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'mood_score': mood_score,
            'tags': tags,
            'journal': journal_input
        })
        
        current_date -= timedelta(days=1)
        print("-" * 20)

    return user_entries[::-1]


def main():
    """Main function to run the application."""
    my_entries = collect_user_entries()

    if my_entries:
        analyzer = MoodAnalyzer(my_entries)
        final_insights = analyzer.run_analysis()

        print("\n" + "="*30)
        print("✨ Your Mindful Metrics Report ✨")
        print("="*30)
        for insight in final_insights:
            icon = {"info": "ℹ️", "positive_pattern": "✅", "negative_pattern": "⚠️",
                    "concerning_trend": "❗️", "care_recommendation": "❤️", "nlp_insight": "🧠"}.get(insight['type'], "🔹")
            print(f"{icon} {insight['text']}\n")

# Run the main application
if __name__ == "__main__":
    main()
