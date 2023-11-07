import pytest
import pandas as pd
import numpy as np

from functions import create_mixup_images

def test_mixup():

    images = [np.array([1,1,1,1]),
    np.array([0,0,0,0]),
    np.array([1,1,1,1]),
    np.array([1,1,1,1])]

    df1 = pd.DataFrame(
        [[0, 0, 0.6, 0.4],
         [0, 1, 0.8, 0.2]],
        columns = ["class", "image_index", "confidence_class_0", "confidence_class_1"]
    )
    df2 = pd.DataFrame(
        [[1, 3, 0.4, 0.6],
         [1, 2, 0.3, 0.7]],
        columns = ["class", "image_index", "confidence_class_0", "confidence_class_1"]
    )
    mixup_df = create_mixup_images(df1,df2,images,1)
    
    # Check image_0 column is correct
    pd.testing.assert_series_equal(mixup_df["image_0"],pd.Series([np.array([1,1,1,1]),np.array([1,1,1,1]),np.array([0,0,0,0]),np.array([0,0,0,0])]),check_names=False)
    # Check image_1 column is correct
    pd.testing.assert_series_equal(mixup_df["image_1"],pd.Series([np.array([1,1,1,1]),np.array([1,1,1,1]),np.array([1,1,1,1]),np.array([1,1,1,1])]),check_names=False)
    # Check length = 4
    assert len(mixup_df) == 4
    print(mixup_df)

    mixup_df = create_mixup_images(df1,df2,images,2)
    # Check length = 8
    assert len(mixup_df) == 8


def test_boundary_points():