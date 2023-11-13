import pytest
import pandas as pd
import numpy as np

from functions import create_mixup_images, find_boundary_points

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
    MUBA_ITERS = 3

    muba_df = pd.DataFrame(
        [[0.1,0.9,np.array([1]),np.array([0]),np.array([0.1]),"mix",1,0],
         [0.4,0.6,np.array([1]),np.array([0]),np.array([0.4]),"mix",1,0],
         [0.8,0.2,np.array([1]),np.array([0]),np.array([0.8]),"mix",1,1],#pred change
         [0.2,0.8,np.array([1]),np.array([0]),np.array([0.2]),"mix",1,0],
         [0.5,0.5,np.array([1]),np.array([0]),np.array([0.5]),"mix",1,1],#pred change
         [0.9,0.1,np.array([1]),np.array([0]),np.array([0.9]),"mix",1,0] #pred change
         ],
         columns = ["alpha_class_0","alpha_class_1","image_0","image_1","mixup_image","type","label","argmax_pred"]
    )

    boundary_df = find_boundary_points(muba_df,MUBA_ITERS)

    expected_df = pd.DataFrame(
        [[0.6,0.4,np.array([1]),np.array([0]),np.array([0.6]),"boundary",0],
         [0.35,0.65,np.array([1]),np.array([0]),np.array([0.35]),"boundary",1],
         [0.7,0.3,np.array([1]),np.array([0]),np.array([0.7]),"boundary",0],
         ],
         columns = ["alpha_class_0","alpha_class_1","image_0","image_1","mixup_image","type","label"]
    )

    pd.testing.assert_frame_equal(boundary_df,expected_df,check_dtype=False)

