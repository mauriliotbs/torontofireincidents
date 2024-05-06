import pandas as pd
import numpy as np
from scipy.stats import spearmanr, chi2_contingency, kruskal

class FeatureAnalysis:

    @staticmethod
    def keepStrongestFeaturesInDataFrame(response_variable, df):
        '''
        response_variable: Name of the response variable (e.g., 'Estimated_Dollar_Loss')
        df: DataFrame containing the dataset
        
        Returns:
        df_cleaned: DataFrame with weakly correlated features eliminated
        '''

        # Calculate Spearman correlation coefficients with p-values
        correlations = df.corr(method='spearman')[response_variable].abs()
        p_values = df.apply(lambda x: spearmanr(x, df[response_variable]).pvalue)

        # Calculate Kruskal-Wallis H-test statistic for each categorical feature
        h_test_stats = {}
        for col in df.select_dtypes(include=['object']):
            h_stat, p_val = kruskal(*[group.values for name, group in df.groupby(col)[response_variable]])
            h_test_stats[col] = h_stat

        # Calculate Chi-Squared test statistic for each categorical feature
        chi2_stats = {}
        for col in df.select_dtypes(include=['object']):
            contingency_table = pd.crosstab(df[col], df[response_variable])
            chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
            chi2_stats[col] = chi2_stat

        # Combine correlation coefficients and p-values into DataFrame
        feature_stats = pd.DataFrame({
            'Correlation': correlations,
            'P-value': p_values,
            'Kruskal-Wallis H-stat': pd.Series(h_test_stats),
            'Chi-Squared Statistic': pd.Series(chi2_stats)
        })

        # Filter features based on correlation significance and strength of association
        significant_features = feature_stats[(feature_stats['P-value'] < 0.05) & (feature_stats['Correlation'] > 0.5)]

        # Drop weakly correlated or insignificant features
        weak_features = set(df.columns) - set(significant_features.index)
        df_cleaned = df.drop(columns=weak_features)

        return df_cleaned
