import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PowerTransformer

class FeatureTransformer:

    NUMERICAL_COLS = []

    CATEGORICAL_COLS = ['Area_of_Origin', 'Building_Status', 'Business_Impact', 'Extent_Of_Fire', 'Final_Incident_Type', 
                    'Fire_Alarm_System_Impact_on_Evacuation', 'Fire_Alarm_System_Operation', 'Fire_Alarm_System_Presence', 
                    'Ignition_Source', 'Initial_CAD_Event_Type', 'Material_First_Ignited', 'Method_Of_Fire_Control', 
                    'Possible_Cause', 'Property_Use', 'Smoke_Alarm_at_Fire_Origin', 'Smoke_Alarm_at_Fire_Origin_Alarm_Failure', 
                    'Smoke_Alarm_at_Fire_Origin_Alarm_Type', 'Smoke_Alarm_Impact_on_Persons_Evacuating_Impact_on_Evacuation', 
                    'Smoke_Spread', 'Sprinkler_System_Operation', 'Sprinkler_System_Presence', 'Status_of_Fire_On_Arrival']

    # ['Business_Impact', 'Extent_Of_Fire', 'Sprinkler_System_Presence', 'Status_of_Fire_On_Arrival']
    ORDINAL_COLS = CATEGORICAL_COLS[2:4] + CATEGORICAL_COLS[20:22]

    # Everything else.
    ONEHOT_COLS = list(set(CATEGORICAL_COLS) - set(ORDINAL_COLS))

    @staticmethod
    def createTransformerPipeline(df):
        # Define categorical and numerical features

        # Get column indexes to fit into ordinal encoder
        #c1_idx = [df.columns.get_loc(item) for item in FeatureTransformer.ORDINAL_COLS]
        c1_idx = [df.columns.get_loc(item) for item in FeatureTransformer.CATEGORICAL_COLS]

        c2_idx = [df.columns.get_loc(item) for item in FeatureTransformer.ONEHOT_COLS]

        response_var_idx = [df.columns.get_loc(item) for item in ['Estimated_Dollar_Loss']]

        # Todo - Numerical features
        # numerical_features = 
        # n_idx = [df.columns.get_loc(item) for item in numerical_features]

        # Create transformers for numerical and categorical features
        # numerical_transformer = Pipeline(steps=[
        #     ('scaler', StandardScaler())
        # ])


                # Custom transformer for log transformation of the response variable
        log_transformer = FunctionTransformer(func=np.log1p, validate=False)


        # Apply transformers to features using ColumnTransformer
        feature_transformer = ColumnTransformer(
            transformers=[
                ('ordinal_encode', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), c1_idx)
                #('onehot_encode', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), c2_idx)
                #('log_transform_response_var', log_transformer, response_var_idx)  # Apply log transformation to response variable
                #('numerical_scaler', numerical_transformer, n_idx), -- todo
            ], remainder='passthrough'  # Keep the remaining columns as they are
            )
            

        return feature_transformer