'''
loader for MIMIC-IV: ready to discharge from ICU prediction
    Key outcome data is the RFD variable:
        0 = Not ready for discharge (ie currently in ICU)    -  1616473   instances
        1 = Successfully discharged (ie went home)           -     7634   instances
        2 = Died                                             -    12243   instances

N.B. MIMIC data is not shipped due to PhysioNet licensing - you must download and
preprocess it yourself (or email Matt for help). By default the loader looks for
the CSV under ``~/datasets/MIMIC-IV``; override with the
``data_path`` argument or the ``MIMIC_DATA_DIR`` environment variable.

# code from https://github.com/jeffnclark/TraCE/blob/main/helpers/funcs_icu_study.py
# paper https://proceedings.mlr.press/v233/clark24a/clark24a.pdf
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>
from __future__ import annotations

from typing import Any

import pandas as pd

from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict
from data_loaders.loaders.external_loaders import mimic_file


class MIMICIVReadyForDischargeLoader(AbstractLoader):
    """Load the MIMIC-IV ready-for-discharge task.

    Binary classification (the 'died' class is dropped): successfully
    discharged (class 1) vs. not ready for discharge / still in ICU (class 0),
    predicted from real-time ICU observations.

    MIMIC data is not shipped (PhysioNet licensing). See the module docstring
    for how to provide it.

    Parameters
    ----------
    data_path : str or None, default=None
        Base directory containing the ``MIMIC-IV/`` folder. When None, falls
        back to the ``MIMIC_DATA_DIR`` env var then the packaged default.
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.1
        Fraction of data used for training in train/test splits.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    desired_variables = ['stay_id',
                         'biocarbonate',
                         'bloodOxygen',
                         'bloodPressure',
                         'bun',
                         'creatinine',
                         'fio2',
                         'haemoglobin',
                         'heartRate',
                         'motorGCS',
                         'eyeGCS',
                         'potassium',
                         'respiratoryRate',
                         'sodium',
                         'Temperature [C]',
                         'verbalGCS',
                         'age',
                         'gender',
                         'hours_since_admission',
                         'RFD']

    def __init__(self,
                 data_path: str | None = None,
                 shuffle: bool = True,
                 train_size: float = 0.1,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='MIMIC-IV Ready for Discharge',
                         **kwargs)
        self.data_path = data_path

    def load_data(self) -> DataDict:
        """Load and preprocess the MIMIC-IV ready-for-discharge CSV.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``.
        """
        filepath = mimic_file(
            'MIMIC-IV', 'full_datatable_timeSeries_Labels.csv', self.data_path)

        df = initial_icu_processing(filepath, self.desired_variables)

        # drop one of the classes (died)
        df = df[df.RFD != 2]

        data: dict[str, Any] = {}
        data['y'] = df.pop('RFD').to_numpy()
        data['X'] = df.to_numpy()
        data['feature_names'] = df.columns.to_list()
        data['label_names'] = ['Not Ready for Discharge', 'Successfully Discharged']
        return data


def nan_post_processing_data(
        df_data: pd.DataFrame,
        columns: list[str],
) -> pd.DataFrame:
    '''
    Function for post processing the dataframe with all the proceessed values (eg removing nans)
    input:
    df_data -  dataframe which contain the values of all the processed files
    columns -  column of interest to perform data processing on
    returns:
    df_data -  post-processed dataframe
    '''
    postprocessSettings = 2

    # for column in df_data:
    for column in columns:
        # determine replacement values
        if postprocessSettings == 0:
            # leave empty field empty
            newValue = ''
        elif postprocessSettings == 1:
            # replace empty field by a zero
            newValue = 0
        elif postprocessSettings == 2:
            # replace empty field with
            # the mean value for column containing numerical values,
            # and the most frequent value in columns
            # with non-numerical/categorical values
            if 'GCS' in column:
                newValue = df_data[column].mode()
                # FOR THE CASE OF THE MODE, NEED TO GET THE FIRST INDEX
                df_data[column] = df_data[column].fillna(newValue[0])
            elif 'gender' in column:
                newValue = df_data[column].mode()
                df_data[column] = df_data[column].fillna(newValue[0])
            else:
                newValue = df_data[column].mean()
                df_data[column] = df_data[column].fillna(newValue)
    return df_data


def change_categorical(
        df: pd.DataFrame,
        categorical_features: list[str],
) -> pd.DataFrame:
    '''
    function: Used to change data variables to categorical type (for the case of passing to DiCE)
    '''
    for column in categorical_features:
        df[column] = df[column].astype('category')
    return df


def initial_icu_processing(
        filepath: str,
        features: list[str],
) -> pd.DataFrame:
    '''
    Function used to perform initial preprocessing of the data, this includes:
    choosing the correct columns of interest,
    changing the label to be negative outocome label from -1 to 2 (for the purpose of DiCE),
    changing the gender label,
    as well as filling in missing values in the data
    input:
        filepath - path of the csv file which cotains the data
        features - which features you want to extract and use

    return:desired_df_data -  dataframe after processing the data
    '''
    df_data = pd.read_csv(filepath,
                          header=0)
    # Get the variables of interest

    # Obtain only the variables of interes
    desired_df_data = df_data[features].copy()
    # Replace values of -1 with 2 (NEGATIVE LABELLED CLASSES DOES NOT SEEM TO WORK WITH DICE)
    desired_df_data['RFD'] = desired_df_data['RFD'].replace(-1, 2)

    # Change gender category from string to float
    desired_df_data['gender'] = desired_df_data['gender'].replace(
        {'M': 0, 'F': 1}).astype('float')

    # Update the mean and the standard deviation for the data
    columns_to_process = desired_df_data.drop(
        columns=['RFD', 'stay_id', 'hours_since_admission']).columns.tolist()
    # columns_to_process=desired_df_data.columns.tolist()

    neutral_data = desired_df_data[desired_df_data['RFD'] == 0]
    positive_data = desired_df_data[desired_df_data['RFD'] == 1]
    negative_data = desired_df_data[desired_df_data['RFD'] == 2]

    # Fill in missing values of the data
    processed_neutral_data = nan_post_processing_data(
        neutral_data, columns_to_process)
    processed_negative_data = nan_post_processing_data(
        negative_data, columns_to_process)
    processed_positive_data = nan_post_processing_data(
        positive_data, columns_to_process)

    desired_df_data.update(processed_neutral_data)
    desired_df_data.update(processed_negative_data)
    desired_df_data.update(processed_positive_data)

    return desired_df_data


# ---------------------------------------------------------------------------
# Backwards-compatible function API (returns pre-split (train, test) dicts)
# ---------------------------------------------------------------------------
def get_ready_for_discharge(
        seed: bool | int = True,
        data_path: str | None = None,
        **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Backwards-compatible helper: return a (train, test) split for RFD."""
    loader = MIMICIVReadyForDischargeLoader(
        data_path=data_path, set_seed=seed, **kwargs)
    return loader.get_train_test_split(seed=seed)
