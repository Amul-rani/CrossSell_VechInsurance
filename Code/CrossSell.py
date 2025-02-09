{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f40b3228-9be0-4210-b259-479fce6f3e7a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import libraries\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "import sklearn\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
    "from sklearn.compose import ColumnTransformer\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "import joblib\n",
    "from imblearn.over_sampling import RandomOverSampler\n",
    "import imblearn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4a7cfbad-329e-490a-9241-d4a439f9e5ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "# read the test train and sample data\n",
    "train = pd.read_csv('../Data/train.csv')\n",
    "test  = pd.read_csv('../Data/test.csv')\n",
    "sub = pd.read_csv('../Data/sample.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "5f03a6d2-216b-4c73-8fba-ec3f339b9cb0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((381109, 12), (127037, 11), (127037, 2))"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# total no.of rows and columns\n",
    "train.shape, test.shape, sub.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "95c4ac71-4702-4cb5-8978-ebd2c31aa17b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "id                        int64\n",
       "Gender                   object\n",
       "Age                       int64\n",
       "Driving_License           int64\n",
       "Region_Code             float64\n",
       "Previously_Insured        int64\n",
       "Vehicle_Age              object\n",
       "Vehicle_Damage           object\n",
       "Annual_Premium          float64\n",
       "Policy_Sales_Channel    float64\n",
       "Vintage                   int64\n",
       "Response                  int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# to view the dataypes\n",
    "train.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d1f9cbb3-5d22-4d41-9109-09996d69e808",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 381109 entries, 0 to 381108\n",
      "Data columns (total 12 columns):\n",
      " #   Column                Non-Null Count   Dtype  \n",
      "---  ------                --------------   -----  \n",
      " 0   id                    381109 non-null  int64  \n",
      " 1   Gender                381109 non-null  object \n",
      " 2   Age                   381109 non-null  int64  \n",
      " 3   Driving_License       381109 non-null  int64  \n",
      " 4   Region_Code           381109 non-null  float64\n",
      " 5   Previously_Insured    381109 non-null  int64  \n",
      " 6   Vehicle_Age           381109 non-null  object \n",
      " 7   Vehicle_Damage        381109 non-null  object \n",
      " 8   Annual_Premium        381109 non-null  float64\n",
      " 9   Policy_Sales_Channel  381109 non-null  float64\n",
      " 10  Vintage               381109 non-null  int64  \n",
      " 11  Response              381109 non-null  int64  \n",
      "dtypes: float64(3), int64(6), object(3)\n",
      "memory usage: 34.9+ MB\n"
     ]
    }
   ],
   "source": [
    "# get all details of the dataset\n",
    "train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c3634ca9-d8d0-4940-bc21-558261130acc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Driving_License</th>\n",
       "      <th>Region_Code</th>\n",
       "      <th>Previously_Insured</th>\n",
       "      <th>Vehicle_Age</th>\n",
       "      <th>Vehicle_Damage</th>\n",
       "      <th>Annual_Premium</th>\n",
       "      <th>Policy_Sales_Channel</th>\n",
       "      <th>Vintage</th>\n",
       "      <th>Response</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Male</td>\n",
       "      <td>44</td>\n",
       "      <td>1</td>\n",
       "      <td>28.0</td>\n",
       "      <td>0</td>\n",
       "      <td>&gt; 2 Years</td>\n",
       "      <td>Yes</td>\n",
       "      <td>40454.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>217</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Male</td>\n",
       "      <td>76</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1-2 Year</td>\n",
       "      <td>No</td>\n",
       "      <td>33536.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>183</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id Gender  Age  Driving_License  Region_Code  Previously_Insured  \\\n",
       "0   1   Male   44                1         28.0                   0   \n",
       "1   2   Male   76                1          3.0                   0   \n",
       "\n",
       "  Vehicle_Age Vehicle_Damage  Annual_Premium  Policy_Sales_Channel  Vintage  \\\n",
       "0   > 2 Years            Yes         40454.0                  26.0      217   \n",
       "1    1-2 Year             No         33536.0                  26.0      183   \n",
       "\n",
       "   Response  \n",
       "0         1  \n",
       "1         0  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# to view the first two rows\n",
    "train.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "40e46f21-1ef5-433e-a4a8-03b9a76b4a8e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "id                      0\n",
       "Gender                  0\n",
       "Age                     0\n",
       "Driving_License         0\n",
       "Region_Code             0\n",
       "Previously_Insured      0\n",
       "Vehicle_Age             0\n",
       "Vehicle_Damage          0\n",
       "Annual_Premium          0\n",
       "Policy_Sales_Channel    0\n",
       "Vintage                 0\n",
       "Response                0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#EDA\n",
    "# checking missing data\n",
    "train.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "20cd3101-8eb1-4656-883c-c99b3f863feb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# to check duplicates\n",
    "train.duplicated().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "e04b058b-58d8-4b5e-ab03-908d935bfc76",
   "metadata": {},
   "outputs": [],
   "source": [
    "tgt_col = ['Response']\n",
    "ign_cols = ['id']\n",
    "cat_cols = train.select_dtypes(include='object').columns\n",
    "num_cols = train.select_dtypes(exclude='object').columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "5f1ffd6f-27bc-4973-9f16-282fe2cea69d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Response']\n",
      "['id']\n",
      "Index(['Gender', 'Vehicle_Age', 'Vehicle_Damage'], dtype='object')\n",
      "Index(['id', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',\n",
      "       'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Response'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "print(tgt_col,ign_cols, cat_cols,num_cols, sep='\\n')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "d428a515-0c78-4633-baa5-e5ec78bbd3ff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Vehicle_Age</th>\n",
       "      <th>Vehicle_Damage</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Male</td>\n",
       "      <td>&gt; 2 Years</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Male</td>\n",
       "      <td>1-2 Year</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Gender Vehicle_Age Vehicle_Damage\n",
       "0   Male   > 2 Years            Yes\n",
       "1   Male    1-2 Year             No"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train[cat_cols].head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "41918197-323e-4d3a-b185-ce984b68ba5b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Driving_License</th>\n",
       "      <th>Region_Code</th>\n",
       "      <th>Previously_Insured</th>\n",
       "      <th>Annual_Premium</th>\n",
       "      <th>Policy_Sales_Channel</th>\n",
       "      <th>Vintage</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>44</td>\n",
       "      <td>1</td>\n",
       "      <td>28.0</td>\n",
       "      <td>0</td>\n",
       "      <td>40454.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>217</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>76</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>33536.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>183</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Age  Driving_License  Region_Code  Previously_Insured  Annual_Premium  \\\n",
       "0   44                1         28.0                   0         40454.0   \n",
       "1   76                1          3.0                   0         33536.0   \n",
       "\n",
       "   Policy_Sales_Channel  Vintage  \n",
       "0                  26.0      217  \n",
       "1                  26.0      183  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "num_cols= train.select_dtypes(exclude='object').drop(columns=ign_cols+tgt_col).columns\n",
    "train[num_cols].head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "6587de26-ed03-4fb0-9a1c-6c92daa76c53",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Response'>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "train['Response'].value_counts().plot(kind='bar')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "7ef43db7-72cc-4e1b-a232-d2830cec1a42",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>id</th>\n",
       "      <td>381109.0</td>\n",
       "      <td>190555.000000</td>\n",
       "      <td>110016.836208</td>\n",
       "      <td>1.0</td>\n",
       "      <td>95278.0</td>\n",
       "      <td>190555.0</td>\n",
       "      <td>285832.0</td>\n",
       "      <td>381109.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>381109.0</td>\n",
       "      <td>38.822584</td>\n",
       "      <td>15.511611</td>\n",
       "      <td>20.0</td>\n",
       "      <td>25.0</td>\n",
       "      <td>36.0</td>\n",
       "      <td>49.0</td>\n",
       "      <td>85.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Driving_License</th>\n",
       "      <td>381109.0</td>\n",
       "      <td>0.997869</td>\n",
       "      <td>0.046110</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Region_Code</th>\n",
       "      <td>381109.0</td>\n",
       "      <td>26.388807</td>\n",
       "      <td>13.229888</td>\n",
       "      <td>0.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>28.0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>52.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Previously_Insured</th>\n",
       "      <td>381109.0</td>\n",
       "      <td>0.458210</td>\n",
       "      <td>0.498251</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Annual_Premium</th>\n",
       "      <td>381109.0</td>\n",
       "      <td>30564.389581</td>\n",
       "      <td>17213.155057</td>\n",
       "      <td>2630.0</td>\n",
       "      <td>24405.0</td>\n",
       "      <td>31669.0</td>\n",
       "      <td>39400.0</td>\n",
       "      <td>540165.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Policy_Sales_Channel</th>\n",
       "      <td>381109.0</td>\n",
       "      <td>112.034295</td>\n",
       "      <td>54.203995</td>\n",
       "      <td>1.0</td>\n",
       "      <td>29.0</td>\n",
       "      <td>133.0</td>\n",
       "      <td>152.0</td>\n",
       "      <td>163.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Vintage</th>\n",
       "      <td>381109.0</td>\n",
       "      <td>154.347397</td>\n",
       "      <td>83.671304</td>\n",
       "      <td>10.0</td>\n",
       "      <td>82.0</td>\n",
       "      <td>154.0</td>\n",
       "      <td>227.0</td>\n",
       "      <td>299.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Response</th>\n",
       "      <td>381109.0</td>\n",
       "      <td>0.122563</td>\n",
       "      <td>0.327936</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                         count           mean            std     min      25%  \\\n",
       "id                    381109.0  190555.000000  110016.836208     1.0  95278.0   \n",
       "Age                   381109.0      38.822584      15.511611    20.0     25.0   \n",
       "Driving_License       381109.0       0.997869       0.046110     0.0      1.0   \n",
       "Region_Code           381109.0      26.388807      13.229888     0.0     15.0   \n",
       "Previously_Insured    381109.0       0.458210       0.498251     0.0      0.0   \n",
       "Annual_Premium        381109.0   30564.389581   17213.155057  2630.0  24405.0   \n",
       "Policy_Sales_Channel  381109.0     112.034295      54.203995     1.0     29.0   \n",
       "Vintage               381109.0     154.347397      83.671304    10.0     82.0   \n",
       "Response              381109.0       0.122563       0.327936     0.0      0.0   \n",
       "\n",
       "                           50%       75%       max  \n",
       "id                    190555.0  285832.0  381109.0  \n",
       "Age                       36.0      49.0      85.0  \n",
       "Driving_License            1.0       1.0       1.0  \n",
       "Region_Code               28.0      35.0      52.0  \n",
       "Previously_Insured         0.0       1.0       1.0  \n",
       "Annual_Premium         31669.0   39400.0  540165.0  \n",
       "Policy_Sales_Channel     133.0     152.0     163.0  \n",
       "Vintage                  154.0     227.0     299.0  \n",
       "Response                   0.0       0.0       1.0  "
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.describe().T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "329f7a46-ba38-4edb-9393-7dc14de931ad",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "id                      381109\n",
       "Gender                       2\n",
       "Age                         66\n",
       "Driving_License              2\n",
       "Region_Code                 53\n",
       "Previously_Insured           2\n",
       "Vehicle_Age                  3\n",
       "Vehicle_Damage               2\n",
       "Annual_Premium           48838\n",
       "Policy_Sales_Channel       155\n",
       "Vintage                    290\n",
       "Response                     2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "1ab31edd-7c5a-4828-a541-1338d3465405",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Vintage', ylabel='Density'>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(train['Vintage'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "bddd267c-880c-4bbf-a6d3-e2820199212f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Age', ylabel='Density'>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkAAAAGwCAYAAABB4NqyAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAABQsklEQVR4nO3deXhTZf428PskaZOu6b7SQssOZS2LRSqiTBFcQHHEFTccURw2eQcQ1BlccNx+DKPAKNswbswIo4xWoSIgSAXZF0spdF9C96Zr2iTn/aNNoLYgtElPknN/rivXRU9Oku9j1dw8qyCKoggiIiIiGVFIXQARERFRV2MAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2VFJXYAjMpvNKCwshI+PDwRBkLocIiIiugaiKKK6uhoRERFQKK7ex8MA1I7CwkJERUVJXQYRERF1QF5eHrp163bVexiA2uHj4wOg+R+gr6+vxNUQERHRtdDr9YiKirJ+j18NA1A7LMNevr6+DEBERERO5lqmr3ASNBEREckOAxARERHJDgMQERERyQ4DEBEREckOAxARERHJDgMQERERyQ4DEBEREckOAxARERHJDgMQERERyQ4DEBEREckOAxARERHJDgMQERERyQ4DEBEREckOAxARERHJDgMQERERyY5K6gLIeXxyMPc373lwdHQXVEJERNQ57AEiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZYQAiIiIi2WEAIiIiItlhACIiIiLZkTwArV69GjExMdBoNIiPj8e+ffuuev/e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2XzpJHdXbTvQfGzFvHnz0L17d3h4eGDMmDH4+eefrc/bre0iOY2JEyeKGzduFE+fPi0eP35cvP3228Xo6GixpqbGes8bb7wh+vj4iFu3bhVPnTolTp8+XQwPDxf1er2ElXfe9u3bxa+//lpMT08X09PTxRdeeEF0c3MTT58+LYqi67b7cocOHRJ79OghDh48WJw7d671uqu2/eWXXxYHDhwoFhUVWR/FxcXW51213eXl5WL37t3Fxx57TDx48KCYlZUlfvfdd+L58+et97hq24uLi1v9vlNSUkQA4u7du0VRdN12v/rqq2JgYKD41VdfiVlZWeJ//vMf0dvbW1y5cqX1HldtuyiK4n333ScOGDBA3Lt3r5iRkSG+/PLLoq+vr5ifny+Kov3azgDkxIqLi0UA4t69e0VRFEWz2SyGhYWJb7zxhvWehoYGUavVimvXrpWqTLvx9/cX161bJ4t2V1dXi7179xZTUlLEcePGWQOQK7f95ZdfFocMGdLuc67c7kWLFoljx4694vOu3PZfmzt3rtizZ0/RbDa7dLtvv/128Yknnmh17Z577hEffvhhURRd+3deV1cnKpVK8auvvmp1fciQIeLSpUvt2nYOgTmxqqoqAEBAQAAAICsrCzqdDklJSdZ71Go1xo0bhwMHDkhSoz2YTCZ89tlnqK2tRUJCgizaPXv2bNx+++2YMGFCq+uu3vaMjAxEREQgJiYG999/PzIzMwG4dru3b9+OESNG4Pe//z1CQkIwbNgwfPjhh9bnXbntl2tsbMRHH32EJ554AoIguHS7x44di127duHcuXMAgBMnTmD//v2YPHkyANf+nRuNRphMJmg0mlbXPTw8sH//fru2nQHISYmiiAULFmDs2LGIi4sDAOshsL8++DU0NLTNAbHO6NSpU/D29oZarcasWbPw3//+FwMGDHD5dn/22Wc4evQoVqxY0eY5V2776NGjsXnzZuzYsQMffvghdDodxowZg7KyMpdud2ZmJtasWYPevXtjx44dmDVrFubMmWM9EsiV2365L774ApWVlXjssccAuHa7Fy1ahAceeAD9+vWDm5sbhg0bhnnz5uGBBx4A4Npt9/HxQUJCAl555RUUFhbCZDLho48+wsGDB1FUVGTXtkt6FAZ13HPPPYeTJ0+2e8aZIAitfhZFsc01Z9S3b18cP34clZWV2Lp1Kx59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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(train['Age'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "dd7e2c3e-4e2b-4ea0-927b-83eae05ae493",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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5z59+ui5555zf5ezw+HQvn37NHbsWN16660+DQgAQHnjVTlPnTpVv/76q2rWrKmcnBx17NhRjRs3VqVKlfTiiy/6OiMAAOWKV685R0RE6KuvvlJycrJSUlLkcrkUFxenbt26+TofAADlTonOnHNycrR06VL375988onS09OVkZGh5cuX68knn9SpU6d8HhIAgPKkRGfO8+fP19KlS9WrVy9J0vTp09WiRQuFhoZKknbs2KFatWrp0Ucf9X1SAADKiRKdOb/zzjsaNmyYx7J3331XycnJSk5O1pQpU/TBBx/4NCAAAOVNicr5hx9+UNOmTd2/h4SEKCDgf5to166dtm3b5rt0AACUQyV6WjsrK0sVKvzvLr/++qvHepfLpdzcXN8kAwCgnCrRmXOdOnW0devWItdv2bJFderUueBQAACUZyUq5x49emj8+PGFXpGdk5OjiRMnqmfPnj4LBwBAeVSip7WfeuopffDBB7r88suVlJSkpk2byuFwaMeOHZo+fbrOnDmjp556qrSyAgBQLpSonCMjI7V27VqNHDlSY8eOlTFG0u8f33njjTdqxowZioyMLJWgAACUFyX+hLAGDRpoxYoVOnz4sHbt2iVJaty4sapWrerzcAAAlEdefXynJFWtWlXt2rXzZRYAACAvv/gCAACUHsoZAADLUM4AAFiGcgYAwDKUMwAAlqGcAQCwDOUMAIBlKGcAACxDOQMAYBnKGQAAy1DOAABYhnIGAMAylDMAAJahnAEAsAzlDACAZShnAAAsQzkDAGAZyhkAAMtQzgAAWIZyBgDAMpQzAACWoZwBALAM5QwAgGUoZwAALEM5AwBgGcr5HPXr19drr73m7xgAgHLMr+WcmJgoh8NR4LZr1y5/xgIAwK8q+DtA9+7dNXfuXI9lNWrU8FMaAAD8z+9PawcHBysqKsrjFhgYqH//+9+66qqrFBISooYNG2rixIk6c+aM+34Oh0MzZ85Ur169dNlll6l58+Zat26ddu3apU6dOiksLEzt27dXWlqa+z5paWnq06ePIiMjValSJbVt21arV68uNl9WVpbuu+8+1axZUxEREerSpYv++9//ltp8AADg93IuzMqVK3XXXXfp4Ycf1rZt2zRz5ky9/fbbevHFFz3GPf/88xo6dKg2b96sZs2aadCgQbr//vs1btw4fffdd5KkpKQk9/jjx4+r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F/t0/wAbVfB3AAD2OnbsmP7+979r+vTpSkhIkCQ1atRI119/vWbNmqWcnBzNnz9fYWFhkqTp06erd+/emjx5com+ZL5Hjx564IEHJEljxozRq6++qjVr1qhZs2aqUaOGJKlatWqKiory8R4CduLMGUCRtm/frtzcXHXt2rXQda1atXIXsyRdd911crlc2rlzZ4ke58orr3T/7HA4FBUVpczMTO+DAxc5yhlAkUJDQ4tcZ4yRw+EodF3+8oCAAJ37rbR5eXkFxgcFBRW4v8vlKmlc4JJBOQMoUpMmTRQaGqpPP/20wLorrrhCmzdv1okTJ9zLvv76awUEBKhp06aSpBo1aujAgQPu9WfPntXWrVtLlKFixYru+wLlBeUMoEghISEaM2aMnnzySc2fP19paWlav369Zs+ercGDByskJEQJCQnaunWrkpOT9dBDD2nIkCHu15u7dOmiZcuWadmyZdqxY4ceeOABHT16tEQZatasqdDQUK1YsUIHDx5UVlZWKewpYBfKGUCxnnnmGT322GMaP368mjdvrjvvvFOZmZm67LLLtHLlSh0+fFht27bVbbfdpq5du2r69Onu+w4bNkwJCQkaOnSoOnbsqAYNGqhz584levwKFSroH//4h2bOnKno6Gj16dPH17sIWMdhzn1BCAAA+BVnzgAAWIZyBgDAMpQzAACWoZwBALAM5QwAgGUoZwAALEM5AwBgGcoZAADLUM4AAFiGcgYAwDKUMwAAlqGcAQCwzP8BpvQQ03R9SHYAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 500x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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GBhoXnjhBeMMgt+D3XrrreaRRx5xaGvdurWZNGmS29edl5dnJJlPP/3U3hYfH2/i4uLKnOfkyZOmevXq5t1337W3HTlyxHh5eZn169cbY4zZt2+fkWS2b99u75OSkmIkmf379xtjLoaNl5eXOXLkiL3PO++8Y3x8fEx+fr7DOqdPn27at29faj3FxcUmLCzMzJkzx97266+/muDgYPPqq69WWs2lGT9+vGnevLkpLi42xnjmvr78zdzT9u+CBQtMcHCw+fXXX+19Zs+ebSSZlStXlrkvjTHmiy++MJIc/qju1q2bGT9+fJnzVEbdnnRclFV3REREqcF/ubi4OBMdHe3QVtn7PDQ01OF9r6oc4xEREfb3jvLgUr+HOnv2rFJTU9W7d2+H9t69e2vbtm1uX39+fr6k//+SoEs2b96skJAQtWzZUg899JDy8vLs01JTU3Xu3DmHmiMiIhQZGWmvOSUlRcHBwbrtttvsfW6//XYFBwc79ImMjHT4IoqYmBidOXNGqampJWo9ePCgIiIi1KxZMw0ePFjfffedJCkrK0u5ubkO9fj4+Khbt272dVVWzb919uxZvfnmmxoxYoRsNptH7+vf8rT9m5KSYr9c/ts+khz2XWny8/Nls9lUq1Yth/a33npL9erVU9u2bTVhwgQVFhbap1VW3Z5yXJRV9+W3TErz448/au3atRo5cmSJaZW5z3/88UdJ//++V1WO8ZycHB06dOgKe9wRX9LjoX7++WdduHBBoaGhDu2hoaHKzc1167qNMXriiSd0xx13KDIy0t5+99136/7771eTJk2UlZWlqVOnKjo6WqmpqfLx8VFubq5q1Kih2rVrl1lzbm6uQkJCSqwzJCTEoc/l2127dm3VqFGjxLbfdtttWr58uVq2bKkff/xRM2bMUJcuXbR3715739L24ffff29f1/Wu+XKrV6/WyZMnlZCQYG/zxH19OU/bv7m5uWratGmJ9UgX76+W5ddff9WkSZM0dOhQhy9RefDBB9WsWTOFhYVpz549mjx5snbt2qWNGzdWWt2edFxcqe6ref311xUYGKiBAwc6tFfmPr+03Pbt29vf96rSMZ6bm6tmzZqVWEdpCH4P99szQOliKF/e5mpjx47V7t279dlnnzm0P/DAA/afIyMj1bFjRzVp0kRr164t8T/wb11ec2n1V6SPdPGN8JJ27dqp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",
      "text/plain": [
       "<Figure size 500x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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vHyZp8ycOVN79uzR9u3bnZaPHz/e8XtcXJzuuOMOdevWTVu3bq33APm5y7M3dDuas83PPfTQQ47f4+PjNXDgQEVHR+udd95xTEhpzphei+yX27Bhgx566CGnv2y9eewb403j3VCWxi77c9XV1ZowYYJqa2u1bt06p3VPPPGE4/e4uDjdcsstuuOOO5STk6N+/fpZmt/b7i/NHX9JeuuttzRp0iQFBQU5LfeW8W/s+bKxy11vj4E6fvNydPv27dWyZct6fyGWlpbW+2vHE55++ml9+OGHSk9PV+fOna+47c0336xu3brp4MGDkqSIiAhdvHhRZ8+eddru59kjIiJ06tSpetf1/fffO21z+e0/e/asqqurmzwGISEhio+P18GDBx2zpK80pt6S/ejRo9q2bZv++Z//+YrbefPYe9t4N7RN3UuzV7pN1dXVGjdunAoLC5WWlvaLX4nXr18/BQQEOP2fWJn/56y8v1xN/i+//FIHDhz4xceDZM34v/HGGw0+X/rKY8BJk9459hF33XWXefLJJ52W9erVy6MTs2pra81TTz1lIiMjzbffftuky5w+fdoEBgaad955xxjz/xMNNm3a5NimuLi4wYkGu3btcmyzc+fOBicaFBcXO7bZuHGjS5ObLly4YDp16mSWLl3qmCSxatUqx/qqqqoGJ0lYnX3x4sUmIiLCVFdXX3E7bxp7NTIxy1vGe926deaGG24wVVVVjm1Wrlx5xYlBFy9eNI899pjp3bu30yz7K8nLy3OanGNl/stZeX+5mvxJSUn1ZqU35lqO/4oVK0ybNm0afb683h4DTeFXJVz3EaUNGzaYgoIC88wzz5iQkBBz5MgRj+3zySefNHa73WRkZDhN+T9//rwxxpjKykozd+5ck5mZaQoLC016eroZOHCg6dSpU70p9507dzbbtm0zOTk5ZujQoQ1Oub/ttttMVlaWycrKMvHx8Q1OuU9MTDQ5OTlm27ZtpnPnzlf8mM/cuXNNRkaGOXz4sNm5c6cZNWqUCQ0NdYzZypUrjd1uN6mpqSYvL89MnDixwY8LWJG9Tk1NjenatatZsGCB03JvHPvKykqTm5trcnNzjSTz6quvmtzcXMfsYW8a77KyMhMeHm4mTpxo8vLyTGpqqgkNDTVz5sxpMH91dbV59NFHTefOnc3u3budHg91T2LfffedWbp0qfn6669NYWGh2bp1q+nZs6e5/fbbLc/vbfcXV/PXKS8vN23atDFvvPGGuZzV4x8QEGCCgoIafb40xvsfA2FhYXxE6Ur+4z/+w3Tr1s20bt3a9OvXz+mjQp4gqcGft99+2xhjzPnz583w4cNNhw4dTEBAgOnatatJSkoyRUVFTtfz448/mpkzZ5p27dqZ4OBgM2rUqHrbnDlzxkyaNMmEhoaa0NBQM2nSJHP27FmnbY4ePWoefvhhExwcbNq1a2dmzpzpNL3+cnWfwQsICDCRkZFmzJgxZu/evY71tbW1jqPMwMBAk5CQYPLy8rwie51PPvnESDIHDhxwWu6NY5+ent7g/SUpKckrx3vPnj3mvvvuM4GBgSYiIsIkJSU1mr+wsLDRx0Pd57aLiopMQkKCadeunWndurWJjo42s2bNqvdZXCvye+P9xZX8ddavX2+Cg4PrffbXG8b/l54vjfH+x8CSJUuafBRsjDF8lSEAABbxm4lZAAB4G0oYAACLUMIAAFiEEgYAwCKUMAAAFqGEAQCwCCUMAIBFKGEAACxCCQNwyZEjR2Sz2bR7926rowDXPUoYAACLUMLAdaa2tlarVq1Sjx49FBgYqK5du+qll16SJOXl5Wno0KEKDg7WTTfdpN/97nc6d+6c47L333+/nnnmGafre+yxxzR16lTHv7t3767ly5dr2rRpCg0NVdeuXfXHP/7Rsb7ui8pvv/122Ww23X///R67rYCvo4SB60xycrJWrVqlRYsWqaCgQP/1X/+l8PBwnT9/Xg8++KBuvPFGff3113r//fe1bds2zZw50+V9vPLKK7rjjjuUm5urGTNm6Mknn9T+/fslSV999ZUkadu2bTp58qRSU1PdevsAf9LK6gAAmq6yslJr1qzR2rVrlZSUJEmKjo7Wvffeq5SUFP3444969913FRISIklau3atHnnkEa1atarpXzIuaeTIkZoxY4YkacGCBVq9erUyMjLUs2dPdejQQZJ00003Ob5kHUDzcCQMXEf27dunqqoqJSYmNriuT58+jgKWpEGDBqm2tlYHDhxwaT+33Xab43ebzaaIiAiVlpY2PziABlHCwHUkODi40XXGGNlstgbX1S1v0aKFLv/20urq6nrbBwQE1Lt8bW2tq3EB/AJKGLiO3HLLLQoODtZnn31Wb11sbKx2796tH374wbFsx44datGihWJiYiRJHTp00MmTJx3ra2pqlJ+f71KG1q1bOy4L4OpQwsB1JCgoSAsWLND8+fP17rvv6tChQ9q5c6c2bNigSZMmKSgoSElJScrPz1d6erqefvppTZ482fF+8NChQ7V161Zt3bpV+/fv14wZM1RWVuZSho4dOyo4OFgff/yxTp06pfLycg/cUsA/UMLAdWbRokWaO3eunn/+efXq1Uvjx49XaWmp2rRpo08++UT/+Mc/dOedd+qf/umflJiYqLVr1zouO23aNCUlJWnKlCkaPHiwoqKiNGTIEJf236pVK7322mtav369IiMjNXr0aHffRMBv2MzlbxABAIBrgiNhAAAsQgkDAGARShgAAItQwgAAWIQSBgDAIpQwAAAWoYQBALAIJQwAgEUoYQAALEIJAwBgEUoYAACLUMIAAFjk/wCqT9FJpu3NAwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 500x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "for col in train.select_dtypes(include='object').columns:\n",
    "    plt.figure(figsize=(5,3))\n",
    "    sns.countplot(y=train[col])\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "932cc406-eb64-4a75-b8d1-06a5fea3e1eb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Gender : Female')"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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AAAAAAoj+gEAAABAYUQ/AAAAACiM6LebbNu2LXPnzs3o0aMzbNiwTJw4MatWrXrd9WvXrs2FF16Y6urqVFdX57Of/Ww2bty4ByeGjqO11+fvfve7nHPOOTnmmGNy7LHHZsqUKXn++ef34MTQsbT2Gv1T//mf/5lDDz00zz777G6eEjqm1l6fr776aq655pqMHj06Rx55ZCZMmJBHH310D04MHUtrr9EXXnghF1xwQY455pgcc8wx+dSnPpXa2to9ODF0TPPmzctpp532F9fsik4k+u0m8+bNy8KFCzNz5szccccdqaioyKRJk1JfX7/D9VOmTMkzzzyTW265JXPnzs1Pf/rTzJgxYw9PDR1Da67PtWvX5swzz8yb3vSm3Hbbbbnhhhuydu3anH322amrq2uD6aF8rf136HbPPfecf3fCbtba63P69Om566678m//9m9ZtGhRevfunUmTJuWPf/zjHp4cOobWXqPTpk1LTU1Nbr755tx8882pra3Neeedt4enho5le/f5a3ZJJ2pgl6urq2s46qijGm6//fbGY6+88krDEUcc0XDfffc1W/+LX/yiYfDgwQ1PPPFE47Gf/OQnDYceemhDbW3tHpkZOorWXp933nlnw9FHH92wefPmxmM1NTUNgwcPbvjZz362R2aGjqS11+h2W7dubfjoRz/a8PGPf7xh8ODBDc8888yeGBc6lNZen7///e8bBg8e3PDDH/6wyfp3vetd/h0Ku0Frr9FXXnmlYfDgwQ2LFy9uPPaDH/ygYfDgwQ0vvfTSHpkZOpLa2tqGs846q+HII49s+Id/+IeGCRMmvO7aXdWJ3Om3G6xcuTIbNmzIyJEjG4/16tUrQ4cOzbJly5qtX758efr165dBgwY1HhsxYkQqKiqyYsWKPTIzdBStvT6PPfbYfO1rX0uXLl2anXvllVd266zQEbX2Gt1u/vz5efXVV3PuuefuiTGhQ2rt9fnQQw+lV69eOe6445qsf/DBB3PsscfukZmhI2ntNdqlS5d0794999xzT9avX5/169fnu9/9bg4++ODss88+e3J06BAefvjh7LPPPrn33nszbNiwv7h2V3Wizjs9La9r+3sgDBgwoMnx/v37p6amptn61atXN1tbWVmZ3r1773A9sPNae30ecMABOeCAA5oc+/rXv54uXbqkurp69w0KHVRrr9Ek+fWvf52bbropd911V1avXr3bZ4SOqrXX59NPP50DDzww3//+93P99ddn9erVGTp0aP71X/+1yf+IAXaN1l6jXbp0yZVXXpkrrrgiw4cPT0VFRfr165fbbrstnTq5Pwh2tbFjx2bs2LEtWrurOpEreTfYtGlTktf+Qv5Uly5ddvgeYJs2bWq29i+tB3Zea6/PP/eNb3wjt99+ey644IL07dt3t8wIHVlrr9GNGzfm05/+dD796U/n4IMP3hMjQofV2utz/fr1+f3vf5958+blggsuyHXXXZfOnTvn1FNPzYsvvrhHZoaOpLXXaENDQx577LEcddRR+da3vpVbb701+++/f84///ysX79+j8wM7Niu6kSi327QtWvXJGn2Zql1dXXp1q3bDtfv6I1V6+rq0r17990zJHRQrb0+t2toaMiXv/zlXHnllTn33HNzxhln7M4xocNq7TU6c+bMHHzwwfnIRz6yR+aDjqy11+fee++dP/7xj5kzZ05GjRqVI444InPmzEmS/Md//MfuHxg6mNZeo9/73vdy++23Z9asWXnHO96RESNGZP78+XnuueeyaNGiPTIzsGO7qhOJfrvB9lsw16xZ0+T4mjVrUlVV1Wx9VVVVs7X19fV5+eWXs99+++2+QaEDau31mSSvvvpqLrroosyfPz8XX3xxLrjggt0+J3RUrb1GFy1alCVLluSoo47KUUcdlUmTJiVJTjzxxHzuc5/b/QNDB7Iz/43buXPnJo/ydu3aNQceeGCeffbZ3TssdECtvUZXrFiRgQMHpkePHo3H9tlnnwwcODBPP/30bp0V+Mt2VScS/XaDIUOGpEePHlm6dGnjsXXr1uWRRx7J8OHDm62vrq5ObW1tVq1a1Xhs+2uPPvro3T8wdCCtvT6T5OKLL85//dd/5ZprrslZZ521p0aFDqm11+j3v//93Hfffbnnnntyzz33ZObMmUmS66+/Pp/61Kf22NzQEbT2+hw+fHi2bNmS3/zmN43HNm/enGeeeSYHHXTQHpkZOpLWXqMDBgzIqlWrmjwquGnTpjz77LOuUWhju6oT+SCP3aCysjITJkzI7Nmz06dPn+y///6ZNWtWqqqqcsIJJ2Tr1q156aWX0rNnz3Tt2jXDhg3L0UcfnWnTpmX69OnZuHFjPv/5z+ekk05ypx/sYq29Pu++++7cf//9ufjiizNixIi88MILjXttXwPsOq29Rv/8f5RsfxPzt7zlLd53E3ax1l6fw4cPzzvf+c5ccsklueKKK9K7d+/MnTs3e+21V/7pn/6prX8dKE5rr9GTTjopCxYsyNSpUxv/j7Ivf/nLqayszAc/+ME2/m2gY9ldncidfrvJlClTcsopp+Tyyy/PRz/60ey1115ZsGBBKisrU1NTk1GjRuX+++9PklRUVOSrX/1qDjjggJx++umZOnVqjjvuuEyfPr1tfwkoVGuuz/vuuy9JcvXVV2fUqFFNvravAXat1lyjwJ7V2uvzK1/5SkaMGJFPfvKTOeWUU7J+/fp84xvfSJ8+fdrwt4ByteYa7d+/f26//fY0NDTk9NNPz5lnnpm999473/72t9OrV682/k2gY9ldnaiioaGhYTfMCwAAAAC0EXf6AQAAAEBhRD8AAAAAKIzoBwAAAACFEf0AAAAAoDCiHwAAAAAURvQDAAAAgMKIfgAAAABQGNEPAAAAAAoj+gEAAABAYUQ/AAAAACiM6AcAAAAAhRH9AAAAAKAwoh8AAAAAFEb0AwAAAIDCiH4AAAAAUBjRDwAAAAAKI/oBAAAAQGFEPwAAAAAojOgHAAAAAIUR/QAAAACgMKIfAAAAABRG9AMAAACAwoh+AAAAAFAY0Q8AAAAACiP6AQAAAEBhRD8AAAAAKIzoBwAAAACFEf0AAAAAoDCiHwAAAAAURvQDAAAAgMKIfgAAAABQGNEPAAAAAAoj+gEAAABAYUQ/AAAAACiM6AcAAAAAhRH9AAAAAKAwoh8AAAAAFEb0AwAAAIDCiH4AAAAAUBjRDwAAAAAKI/oBAAAAQGFEPwAAAAAojOgHAAAAAIUR/QAAAACgMKIfAAAAABRG9AMAAACAwoh+AAAAAFAY0Q8AAAAACiP6AQAAAEBhRD8AAAAAKIzoBwAAAACFEf0AAAAAoDCiHwAAAAAURvQDAAAAgMKIfgAAAABQGNEPAAAAAAoj+gEAAABAYUQ/AAAAACiM6AcAAAAAhenc1gMAANDUk08+mdtvvz0PPfRQamtr07lz5xxyyCH5x3/8x3zoQx/K3nvvvcdmOfTQQ/PJT34ykydP3mM/8089++yzGTdu3Ouef+tb35oHHnhgD070+k477bQkyTe/+c02ngQAQPQDAGhX7r///lx66aX5u7/7u5x55pkZOHBgNm/enB/96Ee56qqr8uMf/zjz589PRUVFW4+6R/3Lv/xLjj/++GbHu3TpsueHAQB4AxD9AADaiSeffDKXXnpp3vnOd+YrX/lKOnf+3/9UGzNmTI455phMmTIl3/ve93LiiSe24aR73lvf+tYceeSRbT0GAMAbhvf0AwBoJ2688cZ06tQpM2fObBL8tnvve9+bk046KZ06/e9/wm3bti3XX399TjjhhBx22GF573vf2+zx0tNOOy2XXXZZrr/++hx//PE5/PDD85GPfCT/8z//02Tdz3/+83z4wx/OsGHD8t73vjc/+9nPms1QV1eXq6++OmPGjMlhhx2WD3zgA7n//vubrBk7dmy+8IUv5PTTT8/RRx+dz33uczv8fZcuXZpDDz00d999d4v/Gf0lL7/8cj73uc/lne98Zw4//PB86EMfypIlS5qsOfTQQ/Ptb387//qv/5p3vOMdGTFiRGbOnJnNmzfnS1/6UkaOHJljjjkml112Werq6hpf99JLL2XGjBl517velcMOOywjRozI+eefn2efffZ152nJ3w0AwO7iTj8AgHZi8eLFGTlyZPr27fu6a770pS81+fP06dNz991359xzz81RRx2VZcuW5Qtf+ELWrVuX888/v3Hd//t//y+DBg3K5ZdfnoaGhnzpS1/KlClT8uCDD2avvfbKww8/nIkTJ+aYY47Jtddem+effz4XXHBBk5/V0NCQ888/P7/4xS8yZcqUDBo0KA888ECmTZuW+vr6nHTSSY1rv/Wtb+VjH/tYzjnnnHTt2nWHv8vb3/723HHHHXnrW9/6V//ZbNu2LVu2bGlyrKKiInvttVeS12Lk6aefnj/84Q+ZNm1a+vfvn0WLFuXss8/OjTfemGOPPbbxdbNnz8773//+fPWrX82DDz6Yb3zjG3nooYcyZMiQzJo1K8uXL8/8+fMzcODAnH322WloaMi5556bV155JRdeeGH69euXRx99NNdee20+97nP5aabbtrhzC39uwEA2B1EPwCAduCVV17JK6+8koMPPrjZudeLXU899VTuvPPOXHDBBTnnnHOSJKNGjUpFRUW+/vWv59RTT82+++7buMeCBQvSo0ePJMmGDRtyySWX5NFHH81hhx2Wr3/96+nTp0+uu+66VFZWJkl69+6dadOmNf7cn/3sZ/nJT36SOXPm5H3ve1+SZPTo0dm0aVNmz56dE088sfEOxf79++df//Vfm9yV+Od69OjR4kd2L7vsslx22WVNju2111555JFHkiTf/e53s3Llytx5550ZNmxYkuS4447LaaedltmzZ2fRokWNrxs0aFCuuOKKJEl1dXXuuuuuvPrqq5k9e3Y6d+6c0aNH58EHH8wvfvGLJMmaNWvSrVu3XHLJJRk+fHiS5Jhjjsmzzz6bhQsX7nDe1vzdAADsDqIfAEA7sG3bth0eX7VqVd7znvc0Obb//vvnwQcfzH//93+noaEhY8eObRIGx44dm+uuuy4rVqzIu9/97iTJIYcc0hj8kmS//fZLkmzatClJsmLFihx//PGNwS9J3vOe9zTeSZckS5YsSUVFRcaMGdPs591777353e9+l7//+79P8lpY+0vBr7U++clPNvsgjz/9MJMlS5akX79+efvb395ktne96125+uqr88orr2SfffZJkhx11FGN5zt37px99903hx12WJNHqnv37p0//vGPSV77Z/WNb3wjSfL8889n1apVefLJJ/OLX/wir7766g7nbc3fDQDA7iD6AQC0A/vuu2+6d++e5557rsnxAQMG5K677mr889e+9rU8/vjjSV57D7skef/737/DPVevXt34fbdu3Zqc2x7ktsfGV155JX369GmyZnsQ2+7ll19OQ0NDjj766B3+vDVr1jRGvze/+c07/kV30v7775/DDz/8dc+//PLLeeGFF/L2t799h+dfeOGFxuj3p/Fzuz//5/Pn7r333vz7v/97ampq0rt37wwZMuR1H1vePk/Ssr8bAIDdQfQDAGgnxo0blwcffDDr169vDFOVlZVNYlfv3r0bv+/Vq1eS5NZbb82b3vSmZvu95S1vafHP7t27d/7whz80OdbQ0JBXXnml8c89e/ZM9+7dG+96+3MHHXRQi3/ertazZ88cfPDBmT179g7PH3DAATu99/Lly3PJJZdkwoQJOeuss1JVVZUkufrqq7NixYodvmZX/t0AAOwMn94LANBOnHvuudm6dWs+85nPpL6+vtn5zZs355lnnmn8c3V1dZJk7dq1Ofzwwxu/Xn755Xz5y19uvNusJY499tj8+Mc/bnzcN0l+8pOfNHl8dcSIEdm4cWMaGhqa/Lzf/e53+drXvtbsvQf3pBEjRqSmpiZ9+/ZtMtuSJUty4403NnlMubV++ctfZtu2bZkyZUpj8Nu6dWvjpxvv6NHsXfl3AwCwM9zpBwDQTrztbW/LNddck0suuSQnnXRSPvShD+XQQw/Nli1b8stf/jJ33XVX/vCHP+Tss89OkgwePDj/+I//mM9+9rN57rnncthhh+Wpp57KnDlzcsABB+zwQ0Fez/nnn58f/OAHOeuss3L22Wdn7dq1mTNnTvbee+/GNWPGjEl1dXXOO++8nHfeeRk0aFB+/etf5ytf+UpGjRrV7PHgv2b9+vV54okn8ta3vrXVr/1zH/zgB3PbbbflzDPPzCc+8YkMGDAgP/vZz3LDDTdkwoQJTX6P1jriiCOSJFdccUXGjx+fdevW5bbbbsvKlSuTJBs3bmz2yPCu/LsBANgZoh8AQDvy7ne/O/fee2++/e1v56677spzzz2XhoaGHHjggXnf+96Xj3zkI02C0VVXXZWvf/3rWbhwYWpra9O3b9+8733vy9SpU1t1d9vBBx+c2267LV/84hczbdq09O3bN5dcckm++MUvNq7p1KlTrr/++lx77bX5+te/nhdffDH77bdfzjjjjJx//vmt/l0ffvjhfPzjH89VV12VD37wg61+/Z/q3r17vvWtb+Waa67JrFmz8sc//jH7779/LrzwwkycOPFv2vuYY47J5z73udx88835r//6r7z5zW/OMccck69+9as5//zzs2LFiowZM6bZ63bV3w0AwM6oaGhoaGjrIQAAAACAXcd7+gEAAABAYUQ/AAAAACiM6AcAAAAAhRH9AAAAAKAwoh8AAAAAFEb0AwAAAIDCdG7rATqq4cOHp76+Pv369WvrUQAAAABoQy+88EIqKyuzfPnyXban6NdG6urqsnXr1rYeAwAAAIA2tmXLljQ0NOzSPUW/NtK/f/8kyeLFi9t4EgAAAADa0rhx43b5nt7TDwAAAAAKI/oBAAAAQGFEPwAAAAAojOgHAAAAAIUR/QAAAACgMKIfAAAAABRG9AMAAACAwoh+AAAAAFAY0Q8AAAAACiP6AQAAAEBhRD8AAAAAKIzoBwAAAACFEf0AAAAAoDCiHwAAAAAURvQDAAAAgMKIfgAAAABQGNEPAAAAAAoj+gEAAABAYUQ/AAAAACiM6AcAAAAAhRH9AAAAAKAwoh8AAAAAFEb0AwAAAIDCiH7sdtu2NbT1CADwhuTfoQAA7KzObT0A5evUqSJfu+uRPPfCxrYeBQDeMPbv1z3nnzK0rccAAOANql1Fv3nz5mXJkiX55je/mSQ57bTT8vOf/3yHa7/0pS/lpJNOynPPPZexY8c2Oz9z5sz88z//c5Lk0UcfzZVXXpnf/va36d27d0477bScddZZjWu3bduWr371q/nOd76TdevW5R3veEc+//nP56CDDmpc89f24C977oWNebpmfVuPAQAAANAhtJvod8stt2Tu3Lmprq5uPPaVr3wlr776apN1l19+eX7/+9/n3e9+d5LkscceS5cuXfKDH/wgFRUVjet69uyZJFm7dm3OPPPMvPvd786MGTPyq1/9KjNmzEjv3r0zfvz4JK/FxoULF+aqq67Kfvvtl1mzZmXSpEm57777UllZ2aI9AAAAAKC9aPPot3r16lx22WVZsWJFBg4c2ORc7969m/z5vvvuy0MPPZS77747PXr0SJI8/vjjGThwYPr377/D/e+8885UVlZm+vTp6dy5cwYNGpRVq1blhhtuyPjx41NfX5+bbropF110UcaMGZMkmTNnTkaPHp0HHngg73//+//qHgAAAADQnrT5B3k8/PDD2WeffXLvvfdm2LBhr7tu48aNufrqq3P66afn0EMPbTz+2GOP5ZBDDnnd1y1fvjzV1dXp3Pl/++bIkSPz1FNP5cUXX8zKlSuzYcOGjBw5svF8r169MnTo0CxbtqxFewAAAABAe9Lmd/qNHTt2h+/J9+cWLlyYDRs25F/+5V+aHH/88cfTr1+/nHrqqXn66adz0EEH5bzzzsvo0aOTJLW1tRk8eHCT12y/K/D5559PbW1tkmTAgAHN1tTU1LRoj759+7b01wUAAACA3a7N7/Rria1bt+ab3/xmTj311Mb36kuS+vr6PP3001m/fn2mTp2a66+/PocffngmTZqUJUuWJEk2b96cysrKJvt16dIlSVJXV5dNmzYlyQ7X1NXVtWgPAAAAAGhP2vxOv5b4+c9/nueffz4f+tCHmhyvrKzMsmXL0rlz58Yod9hhh+XJJ5/MggULcuyxx6Zr166pr69v8rrtoa579+7p2rVrktcC4vbvt6/p1q1bkvzVPQAAAACgPXlD3On3gx/8IEcccUQOPPDAZue6d+/e7C68wYMHZ/Xq1UmSqqqqrFmzpsn57X/eb7/9Gh/r3dGaqqqqFu0BAAAAAO3JGyL6rVixoskHbWy3cuXKHHXUUVm+fHmT47/97W8bP9yjuro6K1asyNatWxvPL1myJAMHDkzfvn0zZMiQ9OjRI0uXLm08v27dujzyyCMZPnx4i/YAAAAAgPak3Ue/rVu35oknnmj2QRrJa3f0ve1tb8uMGTOyfPnyPPnkk7nqqqvyq1/9Kp/4xCeSJOPHj8/69etz2WWX5Yknnsjdd9+dW2+9Neeee26S1x4RnjBhQmbPnp3Fixdn5cqVmTZtWqqqqnLCCSe0aA8AAAAAaE/a/Xv6vfzyy3n11VfTu3fvZuc6deqU+fPnZ/bs2Zk6dWrWrVuXoUOH5uabb86hhx6aJOnbt29uvPHGXHnllTn55JPTr1+/XHzxxTn55JMb95kyZUq2bNmSyy+/PJs3b051dXUWLFjQ+NhwS/YAAAAAgPaioqGhoaGth+iIxo0blyRZvHhxG0+yZ3zmuuV5umZ9W48BAG8YBw/okS/8y/C2HgMAgD1gd3Sidv94LwAAAADQOqIfAAAAABRG9AMAAACAwoh+AAAAAFAY0Q8AAAAACiP6AQAAAEBhRD8AAAAAKIzoBwAAAACFEf0AAAAAoDCiHwAAAAAURvQDAAAAgMKIfgAAAABQGNEPAAAAAAoj+gEAAABAYUQ/AAAAACiM6AcAAAAAhRH9AAAAAKAwoh8AAAAAFEb0AwAAAIDCiH4AAAAAUBjRDwAAAAAKI/oBAAAAQGFEPwAAAAAojOgHAAAAAIUR/QAAAACgMKIfAAAAABRG9AMAAACAwoh+AAAAAFAY0Q8AAAAACiP6AQAAAEBhRD8AAAAAKIzoBwAAAACFEf0AAAAAoDCiHwAAAAAURvQDAAAAgMKIfgAAAABQGNEPAAAAAAoj+gEAAABAYUQ/AAAAACiM6AcAAAAAhRH9AAAAAKAwoh8AAAAAFEb0AwAAAIDCiH4AAAAAUJjObT0AHcNb3ty9rUcAgDcU/+4EAOBvIfqx2zVs25ZP/vPQth4DAN5wGrZtS0UnD2YAANB6oh+7XUWnTvndN76RTatXt/UoAPCG0W2//fK2j3+8rccAAOANSvRjj9i0enU2PvtsW48BAAAA0CF4XgQAAAAACiP6AQAAAEBhRD8AAAAAKIzoBwAAAACFEf0AAAAAoDCiHwAAAAAURvQDAAAAgMKIfgAAAABQGNEPAAAAAAoj+gEAAABAYUQ/AAAAAChMu4p+8+bNy2mnndbk2KWXXppDDz20yddxxx3XeH7btm2ZO3duRo8enWHDhmXixIlZtWpVkz0effTRTJgwIUceeWSOP/74LFiwoMn5XbEHAAAAALQX7Sb63XLLLZk7d26z44899lg+8YlP5KGHHmr8uueeexrPz5s3LwsXLszMmTNzxx13pKKiIpMmTUp9fX2SZO3atTnzzDNz8MEHZ9GiRZk8eXKuvfbaLFq0aJfuAQAAAADtRZtHv9WrV+fss8/Otddem4EDBzY5t3Xr1jzxxBM5/PDD069fv8avPn36JEnq6+tz0003ZfLkyRkzZkyGDBmSOXPmZPXq1XnggQeSJHfeeWcqKyszffr0DBo0KOPHj88ZZ5yRG264YZftAQAAAADtSZtHv4cffjj77LNP7r333gwbNqzJuaeffjp1dXUZNGjQDl+7cuXKbNiwISNHjmw81qtXrwwdOjTLli1LkixfvjzV1dXp3Llz45qRI0fmqaeeyosvvrhL9gAAAACA9qTzX1+ye40dOzZjx47d4bnHH388FRUVufXWW/PjH/84nTp1ypgxYzJ16tT07NkztbW1SZIBAwY0eV3//v1TU1OTJKmtrc3gwYObnU+S559/fpfs0bdv31b/3gAAAACwu7T5nX5/ye9+97t06tQp+++/f+bPn59LLrkkP/rRj3Leeedl27Zt2bRpU5KksrKyyeu6dOmSurq6JMnmzZt3eD5J6urqdskeAAAAANCetPmdfn/J5MmTc8YZZ6RXr15JksGDB6dfv3758Ic/nN/85jfp2rVrktfel2/798lrIa5bt25Jkq5duzZ+IMefnk+S7t2775I9AAAAAKA9add3+lVUVDQGv+22P2ZbW1vb+EjumjVrmqxZs2ZNqqqqkiRVVVU7PJ8k++233y7ZAwAAAADak3Yd/S688MKcddZZTY795je/SZIccsghGTJkSHr06JGlS5c2nl+3bl0eeeSRDB8+PElSXV2dFStWZOvWrY1rlixZkoEDB6Zv3767ZA8AAAAAaE/adfQ78cQT89Of/jTXXXddfv/73+dHP/pRPvOZz+TEE0/MoEGDUllZmQkTJmT27NlZvHhxVq5cmWnTpqWqqionnHBCkmT8+PFZv359LrvssjzxxBO5++67c+utt+bcc89Nkl2yBwAAAAC0J+36Pf3e9a535dprr838+fMzf/789OzZMx/4wAcyderUxjVTpkzJli1bcvnll2fz5s2prq7OggULGj94o2/fvrnxxhtz5ZVX5uSTT06/fv1y8cUX5+STT96lewAAAABAe1HR0NDQ0NZDdETjxo1LkixevLiNJ9kzfj1rVjY++2xbjwEAbxjdDzggR1x0UVuPAQDAHrA7OlG7frwXAAAAAGg90Q8AAAAACiP6AQAAAEBhRD8AAAAAKIzoBwAAAACFEf0AAAAAoDCiHwAAAAAURvQDAAAAgMKIfgAAAABQGNEPAAAAAAoj+gEAAABAYUQ/AAAAACiM6AcAAAAAhRH9AAAAAKAwoh8AAAAAFEb0AwAAAIDCiH4AAAAAUBjRDwAAAAAKI/oBAAAAQGFEPwAAAAAojOgHAAAAAIUR/QAAAACgMKIfAAAAABRG9AMAAACAwoh+AAAAAFAY0Q8AAAAACiP6AQAAAEBhRD8AAAAAKIzoBwAAAACFEf0AAAAAoDCiHwAAAAAURvQDAAAAgMKIfgAAAABQGNEPAAAAAAoj+gEAAABAYUQ/AAAAACiM6AcAAAAAhRH9AAAAAKAwoh8AAAAAFEb0AwAAAIDCiH4AAAAAUBjRDwAAAAAKI/oBAAAAQGFEPwAAAAAojOgHAAAAAIUR/QAAAACgMKIfAAAAABRG9AMAAACAwoh+AAAAAFAY0Q8AAAAACiP6AQAAAEBhRD8AAAAAKIzoBwAAAACFEf0AAAAAoDCiHwAAAAAURvQDAAAAgMKIfgAAAABQGNEPAAAAAArTrqLfvHnzctpppzU59uCDD2b8+PE56qijMnbs2HzpS1/K5s2bG88/99xzOfTQQ5t9fec732lc8+ijj2bChAk58sgjc/zxx2fBggVNfsa2bdsyd+7cjB49OsOGDcvEiROzatWqJmv+2h4AAAAA0F60m+h3yy23ZO7cuU2OLV++PJ/85Cfz3ve+N/fcc0+mT5+e//t//29mzJjRuOaxxx5Lly5d8pOf/CQPPfRQ49cHPvCBJMnatWtz5pln5uCDD86iRYsyefLkXHvttVm0aFHjHvPmzcvChQszc+bM3HHHHamoqMikSZNSX1/f4j0AAAAAoL3o3NYDrF69OpdddllWrFiRgQMHNjm3cOHCjBw5Muecc06S5KCDDsq0adPymc98JjNmzEhlZWUef/zxDBw4MP3799/h/nfeeWcqKyszffr0dO7cOYMGDcqqVatyww03ZPz48amvr89NN92Uiy66KGPGjEmSzJkzJ6NHj84DDzyQ97///X91DwAAAABoT9r8Tr+HH344++yzT+69994MGzasybmJEyfm4osvbvaaLVu2ZP369Uleu9PvkEMOed39ly9fnurq6nTu/L99c+TIkXnqqafy4osvZuXKldmwYUNGjhzZeL5Xr14ZOnRoli1b1qI9AAAAAKA9afM7/caOHZuxY8fu8NzQoUOb/Lm+vj4333xz3v72t6dPnz5Jkscffzz9+vXLqaeemqeffjoHHXRQzjvvvIwePTpJUltbm8GDBzfZZ/tdgc8//3xqa2uTJAMGDGi2pqampkV79O3bt9W/NwAAAADsLm1+p19LbdmyJRdffHGeeOKJfP7zn0/yWgR8+umns379+kydOjXXX399Dj/88EyaNClLlixJkmzevDmVlZVN9urSpUuSpK6uLps2bUqSHa6pq6tr0R4AAAAA0J60+Z1+LbE96i1dujRz585tfAy4srIyy5YtS+fOnRuj3GGHHZYnn3wyCxYsyLHHHpuuXbs2fiDHdttDXffu3dO1a9ckrwXE7d9vX9OtW7ck+at7AAAAAEB70u7v9FuzZk0+9rGP5Ze//GVuuOGGZo8Cd+/evdldeIMHD87q1auTJFVVVVmzZk2zPZNkv/32a3ysd0drqqqqWrQHAAAAALQn7Tr6vfLKKzn99NPz0ksv5fbbb2/yYRtJsnLlyhx11FFZvnx5k+O//e1vGz/co7q6OitWrMjWrVsbzy9ZsiQDBw5M3759M2TIkPTo0SNLly5tPL9u3bo88sgjGT58eIv2AAAAAID2pF1Hv6uuuirPPPNMZs2alT59+uSFF15o/Nq6dWsGDx6ct73tbZkxY0aWL1+eJ598MldddVV+9atf5ROf+ESSZPz48Vm/fn0uu+yyPPHEE7n77rtz66235txzz03y2iPCEyZMyOzZs7N48eKsXLky06ZNS1VVVU444YQW7QEAAAAA7Um7fU+/bdu25f7778+rr76a008/vdn5xYsX54ADDsj8+fMze/bsTJ06NevWrcvQoUNz880359BDD02S9O3bNzfeeGOuvPLKnHzyyenXr18uvvjinHzyyY17TZkyJVu2bMnll1+ezZs3p7q6OgsWLGh8bLglewAAAABAe1HR0NDQ0NZDdETjxo1L8lq87Ah+PWtWNj77bFuPAQBvGN0POCBHXHRRW48BAMAesDs6Ubt+vBcAAAAAaD3RDwAAAAAKI/oBAAAAQGFEPwAAAAAojOgHAAAAAIUR/QAAAACgMKIfAAAAABRG9AMAAACAwoh+AAAAAFAY0Q8AAAAACiP6AQAAAEBhRD8AAAAAKIzoBwAAAACFEf0AAAAAoDCiHwAAAAAURvQDAAAAgMKIfgAAAABQGNEPAAAAAAoj+gEAAABAYUQ/AAAAACiM6AcAAAAAhRH9AAAAAKAwoh8AAAAAFEb0AwAAAIDCiH4AAAAAUBjRDwAAAAAKI/oBAAAAQGFEPwAAAAAojOgHAAAAAIUR/QAAAACgMKIfAAAAABRG9AMAAACAwoh+AAAAAFAY0Q8AAAAACiP6AQAAAEBhRD8AAAAAKIzoBwAAAACFEf0AAAAAoDCiHwAAAAAURvQDAAAAgMKIfgAAAABQGNEPAAAAAAoj+gEAAABAYUQ/AAAAACiM6AcAAAAAhRH9AAAAAKAwoh8AAAAAFEb0AwAAAIDCiH4AAAAAUBjRDwAAAAAKI/oBAAAAQGFEPwAAAAAojOgHAAAAAIUR/QAAAACgMKIfAAAAABRG9AMAAACAwoh+AAAAAFAY0Q8AAAAACtOuot+8efNy2mmnNTn26KOPZsKECTnyyCNz/PHHZ8GCBU3Ob9u2LXPnzs3o0aMzbNiwTJw4MatWrdrjewAAAABAe9Fuot8tt9ySuXPnNjm2du3anHnmmTn44IOzaNGiTJ48Oddee20WLVrUuGbevHlZuHBhZs6cmTvuuCMVFRWZNGlS6uvr9+geAAAAANBedG7rAVavXp3LLrssK1asyMCBA5ucu/POO1NZWZnp06enc+fOGTRoUFatWpUbbrgh48ePT319fW666aZcdNFFGTNmTJJkzpw5GT16dB544IG8//3v3yN7AAAAAEB70uZ3+j388MPZZ599cu+992bYsGFNzi1fvjzV1dXp3Pl/2+TIkSPz1FNP5cUXX8zKlSuzYcOGjBw5svF8r169MnTo0CxbtmyP7QEAAAAA7clORb9ly5Zlw4YNOzy3bt26fO9732vxXmPHjs0111yTAw88sNm52traVFVVNTnWv3//JMnzzz+f2traJMmAAQOarampqdljewAAAABAe7JT0e/jH/94nnzyyR2ee+SRR3LppZf+TUNtt3nz5lRWVjY51qVLlyRJXV1dNm3alCQ7XFNXV7fH9gAAAACA9qTF7+l3ySWXNN751tDQkOnTp6dHjx7N1j399NN585vfvEuG69q1a+OHaWy3PbJ17949Xbt2TZLU19c3fr99Tbdu3fbYHgAAAADQnrT4Tr/3vve9aWhoSENDQ+Ox7X/e/tWpU6cceeSRueqqq3bJcFVVVVmzZk2TY9v/vN9++zU+krujNdsfx90TewAAAABAe9LiO/3Gjh2bsWPHJklOO+20TJ8+PYMGDdptgyVJdXV1Fi5cmK1bt2avvfZKkixZsiQDBw5M375907Nnz/To0SNLly7NW9/61iSvvafgI488kgkTJuyxPQAAAACgPdmp9/T75je/uduDX5KMHz8+69evz2WXXZYnnngid999d2699dace+65SV57H74JEyZk9uzZWbx4cVauXJlp06alqqoqJ5xwwh7bAwAAAADakxbf6fenNm3alPnz5+eHP/xhNm3alG3btjU5X1FRkR/84Ad/83B9+/bNjTfemCuvvDInn3xy+vXrl4svvjgnn3xy45opU6Zky5Ytufzyy7N58+ZUV1dnwYIFjR+8saf2AAAAAID2oqLhT9+kr4Uuv/zyLFq0KCNGjEhVVVU6dWp+w+Cuel+/Uo0bNy5Jsnjx4jaeZM/49axZ2fjss209BgC8YXQ/4IAccdFFbT0GAAB7wO7oRDt1p9/3v//9TJs2Leecc84uGwQAAAAA2DV26j39tmzZkiOOOGJXzwIAAAAA7AI7Ff1GjRqVH//4x7t6FgAAAABgF9ipx3vf97735fOf/3xeeumlDBs2LN26dWu25qSTTvpbZwMAAAAAdsJORb+pU6cmSe65557cc889zc5XVFSIfgAAAADQRnYq+nWUT5wFAAAAgDeinYp++++//66eAwAAAADYRXYq+n31q1/9q2s++clP7szWAAAAAMDfaJdHvx49eqR///6iHwAAAAC0kZ2KfitXrmx2bOPGjVmxYkWmT5+ez372s3/zYAAAAADAzum0qzbq3r17Ro8enfPPPz9XX331rtoWAAAAAGilXRb9thswYECefPLJXb0tAAAAANBCO/V47440NDSkpqYmN9xwg0/3BQAAAIA2tFPRb8iQIamoqNjhuYaGBo/3AgAAAEAb2qnod/755+8w+vXo0SPHH398Dj744L91LgAAAABgJ+1U9Js8efKungMAAAAA2EV2+j396uvrc/fdd2fp0qVZt25d9t133wwfPjwnn3xyunTpsitnBAAAAABaYaei37p16/Lxj388K1euzFve8pb069cvTz31VO67775861vfyu23356ePXvu6lkBAAAAgBbotDMvuuaaa1JbW5vbbrstDz74YO644448+OCDue222/Liiy/m2muv3dVzAgAAAAAttFPRb/HixZk6dWqGDx/e5Pjw4cMzZcqUfP/7398lwwEAAAAArbdT0W/Dhg058MADd3juwAMPzMsvv/y3zAQAAAAA/A12Kvr93d/9XX74wx/u8NzixYtz0EEH/U1DAQAAAAA7b6c+yOOss87KBRdckPr6+nzgAx/Im9/85vzhD3/If/7nf+Y73/lOpk+fvovHBAAAAABaaqei3/ve9748/fTTmT9/fr7zne80Ht97771z/vnn58Mf/vAuGxAAAAAAaJ2din4bN27MeeedlwkTJuRXv/pVXnnlldTU1OTDH/5w9tlnn109IwAAAADQCq16T79HH300J510Um655ZYkSa9evXLcccfluOOOy5e//OWceuqpefLJJ3fHnAAAAABAC7U4+j3zzDM544wz8sorr+SQQw5pcq6ysjKf+cxnsmHDhpx66qmpra3d5YMCAAAAAC3T4uh3/fXXZ999981//Md/5D3veU+Tc926dcuECROyaNGidO/ePfPnz9/lgwIAAAAALdPi6LdkyZKcffbZ6d279+uu6du3b84888wsWbJkV8wGAAAAAOyEFke/F154IQcddNBfXTd48GCP9wIAAABAG2px9OvTp0/WrFnzV9e99NJLf/FuQAAAAABg92px9Kuurs7dd9/9V9fdc889+fu///u/aSgAAAAAYOe1OPqddtppWbp0ab74xS+mrq6u2fn6+vp86Utfyk9+8pN87GMf26VDAgAAAAAt17mlCw8//PBceuml+cIXvpDvfve7OfbYY3PAAQdk69atef7557N06dKsXbs2n/rUpzJ69OjdOTMAAAAA8Be0OPolycc+9rEMGTIkCxYsyOLFixvv+HvTm96UUaNGZeLEiRk2bNhuGRQAAAAAaJlWRb8kecc73pF3vOMdSZK1a9emU6dO2WeffXb5YAAAAADAzml19PtT++67766aAwAAAADYRVr8QR4AAAAAwBuD6AcAAAAAhRH9AAAAAKAwoh8AAAAAFEb0AwAAAIDCiH4AAAAAUBjRDwAAAAAKI/oBAAAAQGFEPwAAAAAojOgHAAAAAIUR/QAAAACgMKIfAAAAABRG9AMAAACAwoh+AAAAAFAY0Q8AAAAACiP6AQAAAEBhRD8AAAAAKIzoBwAAAACFEf0AAAAAoDCiHwAAAAAURvQDAAAAgMKIfgAAAABQmHYf/ZYuXZpDDz10h1/jxo1Lklx66aXNzh133HGNe2zbti1z587N6NGjM2zYsEycODGrVq1q8nMeffTRTJgwIUceeWSOP/74LFiwoMn5luwBAAAAAO1B57Ye4K856qij8tBDDzU59vjjj+ecc87JJz7xiSTJY489lk984hOZMGFC45q99tqr8ft58+Zl4cKFueqqq7Lffvtl1qxZmTRpUu67775UVlZm7dq1OfPMM/Pud787M2bMyK9+9avMmDEjvXv3zvjx41u0BwAAAAC0F+3+Tr/Kysr069ev8at379656qqr8p73vCf//M//nK1bt+aJJ57I4Ycf3mRdnz59kiT19fW56aabMnny5IwZMyZDhgzJnDlzsnr16jzwwANJkjvvvDOVlZWZPn16Bg0alPHjx+eMM87IDTfc0OI9AAAAAKC9aPfR789961vfSk1NTS699NIkydNPP526uroMGjRoh+tXrlyZDRs2ZOTIkY3HevXqlaFDh2bZsmVJkuXLl6e6ujqdO//vjY8jR47MU089lRdffLFFewAAAABAe9HuH+/9U3V1dZk/f35OP/309O/fP8lrj/pWVFTk1ltvzY9//ON06tQpY8aMydSpU9OzZ8/U1tYmSQYMGNBkr/79+6empiZJUltbm8GDBzc7nyTPP/98i/YAAAAAgPbiDXWn33e/+93U1dXltNNOazz2u9/9Lp06dcr++++f+fPn55JLLsmPfvSjnHfeedm2bVs2bdqUJM3ed69Lly6pq6tLkmzevHmH55PXQmNL9gAAAACA9uINdaffPffck/e85z3Zd999G49Nnjw5Z5xxRnr16pUkGTx4cPr165cPf/jD+c1vfpOuXbsmee19+bZ/n7wW87p165Yk6dq1a+rr65v8rO0xr3v37i3aAwAAAADaizfMnX4vvfRSfvnLX+Z973tfk+MVFRWNwW+77Y/q1tbWNj6Su2bNmiZr1qxZk6qqqiRJVVXVDs8nyX777deiPQAAAACgvXjDRL9f/OIXqaioyIgRI5ocv/DCC3PWWWc1Ofab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zkc/+tFMnDgxV199dR577LFOx0yfPj1r167N3/3d3+WKK67IRRddlJtuuin79u3r8jxJ8uqrr2bJkiW59NJLM2nSpHzmM5/J7t27f89PDwCg5xD9AAB6oY6Ojhw9erT0aG1tTUNDQ5YtW5bDhw/n2muvTZJ8/etfz5e+9KVcfvnlWbduXa6//vo88MADueOOO0prLV26NC+88ELuvPPO3H///amurs4tt9ySHTt2dHrN+fPnZ8aMGfna176WMWPGZMmSJdm+fXuS5MiRI/n0pz+drVu3Zu7cuamvr88ll1yS22+/PevWreu0zje/+c38+te/zt13350VK1bkueeey9/8zd90eZ7f/OY3+eQnP5mf//zn+dKXvpTVq1envb09119/ffbu3fsH+bwBAE42l/cCAPRCO3fuzAUXXNBpW0VFRcaPH5+6urpMnz49b731Vu6777584hOfyLJly5IkH/nIR3LmmWdm2bJlufHGG3Peeefl6aefzuc///l89KMfTZJcdtllOfPMM3Paaad1Wn/27NlZsGBBkmTq1KmZNWtW6uvrM2PGjGzZsiXPP/98HnrooVxyySWlY44ePZr6+vp88pOfzJlnnpkkGTJkSOrr60vrv/TSS7n33ntz4MCBDB069D3n2bhxY95444380z/9U0aNGpUkmTZtWv78z/88dXV1Wbt2bXd/3AAAJ53oBwDQC11wwQW58847kyRNTU2pq6vL22+/nTVr1mTcuHFJkn/9139NS0tLpk+f3uly3unTpydJnnzyyZx33nm57LLLcu+992bPnj2pra3NtGnTcsstt5zwmsfPHkzeCYwzZ87Mvffem5aWljz99NMZNWpUKfgd97GPfSyPPfZYnn322dTW1iZJJk6c2CkoVlVVJUlaWloydOjQ95znqaeeyh//8R9nxIgRpffVp0+fTJs2LVu3bv3dP1QAgB5E9AMA6IU+8IEPZOLEiUneiWgf/vCHc+2112bu3Ln59re/nbPOOitvvPFGkuSv/uqv3nWN5ubmJMmaNWuybt26fP/738/jjz+ePn365Iorrsjy5cvzoQ99qHT8iBEjOj1/2LBh6ejoyFtvvZU333wzH/zgB094jePbDh48WNo2cODATsf06fPOHWva29u7NM8bb7yRhoaGE850PK6lpeWE1wAAONWIfgAAZNiwYbnjjjuycOHC3HXXXVm9enWGDBmSJFm1alVGjx59wnOOB7nBgwdn6dKlWbp0aX79619n+/btqa+vz5133pkHH3ywdPyBAwc6hb///M//zGmnnZYzzzwzZ5xxRhoaGk54jddeey1JMnTo0C6/l/eaZ/Dgwbn00ktz8803v+vzKysru/xaAAA9lS/yAAAgSfJnf/ZnmTp1ar773e9mx44dmTRpUvr165empqZMnDix9OjXr19Wr16dl19+Oa+88kpqa2vz+OOPJ0nGjh2befPm5YorrkhjY2On9X/84x+Xfu7o6MgPf/jDXHLJJamsrExNTU1eeeWVPPPMM52es3Xr1vTr1y8XXXRRl95DV+a59NJLs2/fvowZM6bT+9q6dWseffTRE+5FCABwKnKmHwAAJbfddls+9rGPZcWKFfn2t7+dz372s6mrq8uhQ4dy2WWXle7/V1FRkQkTJmTw4MGpqqrKihUrcujQofzRH/1RnnvuuTzxxBOZP39+p7VXrlyZtra2jBkzJo8++mj27t2bjRs3Jkn+4i/+Ig899FAWLFiQRYsW5UMf+lB+/OMfZ/PmzVmwYEHprMP3MmrUqPecZ86cOfnnf/7nzJkzJ3Pnzs3QoUPzve99L4888khuvfXW7v1AAQDKRPQDAKBk7NixueGGG7Jhw4Z861vfyuLFizN8+PA89NBDefDBB3PGGWfk8ssvz5IlSzJ48OAkyde+9rXcc889qaury4EDBzJy5MgsWLDghHsBLl++PF//+tfzH//xH6murs6GDRsyefLkJO/cp+8f//Efs3r16qxduzaHDh3K2LFjc9ddd+XjH//4+3oP7zXPiBEjsmnTpqxevTrLly9Pa2trRo8e/Tu9FgBAT1XR0dHRUe4hAAAori1btuTWW2/N9u3bc84555R7HACAXsE9/QAAAACgYEQ/AAAAACgYl/cCAAAAQME40w8AAAAACkb0AwAAAICCEf0AAAAAoGBEPwAAAAAoGNEPAAAAAApG9AMAAACAghH9AAAAAKBgRD8AAAAAKBjRDwAAAAAK5v8DoizlGKD6cK0AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 1500x1500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set(style=\"white\", palette=\"muted\", color_codes=True)\n",
    "\n",
    "f, axes = plt.subplots(2, 1, figsize=(15, 15))\n",
    "\n",
    "male = train[train['Gender'] =='Male'][\"Response\"].value_counts().rename('Count')\n",
    "\n",
    "female = train[train['Gender'] =='Female'][\"Response\"].value_counts().rename('Count')\n",
    "\n",
    "sns.barplot(x=male.index,y=male,color=\"b\").set_title('Gender : Male')\n",
    "sns.barplot(x=female.index,y=female,color=\"r\").set_title('Gender : Female')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "f4ed673d-4d68-40a6-827b-feeaca83fa3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#create pipeline to do preprocessing of category columns\n",
    "cat_pipe_encode = Pipeline(\n",
    "    steps = [\n",
    "        ('impute_cat',SimpleImputer(strategy='most_frequent')), #missing values\n",
    "        ('ohe',OneHotEncoder(handle_unknown='ignore')) # Category encoding\n",
    "    ]\n",
    ")\n",
    "num_pipe_encode = Pipeline(\n",
    "    steps = [\n",
    "        ('impute_num',SimpleImputer(strategy='median')), #missing values\n",
    "        ('scale',StandardScaler()) # std scaler\n",
    "    ]\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "fe340f07-8d38-48c3-b137-598b70177b3f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# create map b/w the pipeline and the columns\n",
    "preprocess = ColumnTransformer(\n",
    "    transformers =[\n",
    "        ('cat_encode',cat_pipe_encode, cat_cols), # Cat cols\n",
    "        ('num_encode',num_pipe_encode, num_cols) #numerical cols\n",
    "    ]\n",
    ")\n",
    "# create object for the LogisticRegression\n",
    "mymodel = LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "d2336c02-d2c6-402d-bd28-8bfe464cf3fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# merging the preprocessing  and modelling in a pipeline\n",
    "model_pipeline = Pipeline(\n",
    "    steps=[\n",
    "        ('preprocess', preprocess), #preprocessinng\n",
    "        ('model',mymodel) #modeling\n",
    "    ]\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "7ff521fe-53ae-4ec2-97ad-4f1c45113812",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Driving_License</th>\n",
       "      <th>Region_Code</th>\n",
       "      <th>Previously_Insured</th>\n",
       "      <th>Vehicle_Age</th>\n",
       "      <th>Vehicle_Damage</th>\n",
       "      <th>Annual_Premium</th>\n",
       "      <th>Policy_Sales_Channel</th>\n",
       "      <th>Vintage</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Male</td>\n",
       "      <td>44</td>\n",
       "      <td>1</td>\n",
       "      <td>28.0</td>\n",
       "      <td>0</td>\n",
       "      <td>&gt; 2 Years</td>\n",
       "      <td>Yes</td>\n",
       "      <td>40454.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>217</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Male</td>\n",
       "      <td>76</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1-2 Year</td>\n",
       "      <td>No</td>\n",
       "      <td>33536.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>183</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Gender  Age  Driving_License  Region_Code  Previously_Insured Vehicle_Age  \\\n",
       "0   Male   44                1         28.0                   0   > 2 Years   \n",
       "1   Male   76                1          3.0                   0    1-2 Year   \n",
       "\n",
       "  Vehicle_Damage  Annual_Premium  Policy_Sales_Channel  Vintage  \n",
       "0            Yes         40454.0                  26.0      217  \n",
       "1             No         33536.0                  26.0      183  "
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = train.drop(columns = ign_cols+ tgt_col)\n",
    "X.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "00afe8dc-ab81-4698-bbc8-84d101fdaab2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Response</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Response\n",
       "0         1\n",
       "1         0"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = train[tgt_col]\n",
    "y.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "379057e8-cc50-4567-9fe0-fa062b19aa65",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_X, val_X, train_y,val_y = train_test_split(X,y,test_size=0.1, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "8fbba7b6-f519-475f-bd5c-788c56d8fe88",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((342998, 10), (38111, 10), (342998, 1), (38111, 1))"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_X.shape, val_X.shape,train_y.shape, val_y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "f24b73de-e8d1-41e6-a72d-839a69d87441",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((381109, 12), 342998, 38110)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.shape, int(train.shape[0]*.9), int(train.shape[0]*.1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "ce8ea361-db8d-457f-89d9-ae1c3e8f80e1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                 ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                  (&#x27;ohe&#x27;,\n",
       "                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                  Index([&#x27;Gender&#x27;, &#x27;Vehicle_Age&#x27;, &#x27;Vehicle_Damage&#x27;], dtype=&#x27;object&#x27;)),\n",
       "                                                 (&#x27;num_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                                  (&#x27;scale&#x27;,\n",
       "                                                                   StandardScaler())]),\n",
       "                                                  Index([&#x27;Age&#x27;, &#x27;Driving_License&#x27;, &#x27;Region_Code&#x27;, &#x27;Previously_Insured&#x27;,\n",
       "       &#x27;Annual_Premium&#x27;, &#x27;Policy_Sales_Channel&#x27;, &#x27;Vintage&#x27;],\n",
       "      dtype=&#x27;object&#x27;))])),\n",
       "                (&#x27;model&#x27;, LogisticRegression())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" ><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">Pipeline</label><div class=\"sk-toggleable__content\"><pre>Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                 ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                  (&#x27;ohe&#x27;,\n",
       "                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                  Index([&#x27;Gender&#x27;, &#x27;Vehicle_Age&#x27;, &#x27;Vehicle_Damage&#x27;], dtype=&#x27;object&#x27;)),\n",
       "                                                 (&#x27;num_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                                  (&#x27;scale&#x27;,\n",
       "                                                                   StandardScaler())]),\n",
       "                                                  Index([&#x27;Age&#x27;, &#x27;Driving_License&#x27;, &#x27;Region_Code&#x27;, &#x27;Previously_Insured&#x27;,\n",
       "       &#x27;Annual_Premium&#x27;, &#x27;Policy_Sales_Channel&#x27;, &#x27;Vintage&#x27;],\n",
       "      dtype=&#x27;object&#x27;))])),\n",
       "                (&#x27;model&#x27;, LogisticRegression())])</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" ><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">preprocess: ColumnTransformer</label><div class=\"sk-toggleable__content\"><pre>ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                 Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                 (&#x27;ohe&#x27;,\n",
       "                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                 Index([&#x27;Gender&#x27;, &#x27;Vehicle_Age&#x27;, &#x27;Vehicle_Damage&#x27;], dtype=&#x27;object&#x27;)),\n",
       "                                (&#x27;num_encode&#x27;,\n",
       "                                 Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                  SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                 (&#x27;scale&#x27;, StandardScaler())]),\n",
       "                                 Index([&#x27;Age&#x27;, &#x27;Driving_License&#x27;, &#x27;Region_Code&#x27;, &#x27;Previously_Insured&#x27;,\n",
       "       &#x27;Annual_Premium&#x27;, &#x27;Policy_Sales_Channel&#x27;, &#x27;Vintage&#x27;],\n",
       "      dtype=&#x27;object&#x27;))])</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" ><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">cat_encode</label><div class=\"sk-toggleable__content\"><pre>Index([&#x27;Gender&#x27;, &#x27;Vehicle_Age&#x27;, &#x27;Vehicle_Damage&#x27;], dtype=&#x27;object&#x27;)</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" ><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" ><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">OneHotEncoder</label><div class=\"sk-toggleable__content\"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div></div></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-6\" type=\"checkbox\" ><label for=\"sk-estimator-id-6\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">num_encode</label><div class=\"sk-toggleable__content\"><pre>Index([&#x27;Age&#x27;, &#x27;Driving_License&#x27;, &#x27;Region_Code&#x27;, &#x27;Previously_Insured&#x27;,\n",
       "       &#x27;Annual_Premium&#x27;, &#x27;Policy_Sales_Channel&#x27;, &#x27;Vintage&#x27;],\n",
       "      dtype=&#x27;object&#x27;)</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-7\" type=\"checkbox\" ><label for=\"sk-estimator-id-7\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-8\" type=\"checkbox\" ><label for=\"sk-estimator-id-8\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">StandardScaler</label><div class=\"sk-toggleable__content\"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-9\" type=\"checkbox\" ><label for=\"sk-estimator-id-9\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "Pipeline(steps=[('preprocess',\n",
       "                 ColumnTransformer(transformers=[('cat_encode',\n",
       "                                                  Pipeline(steps=[('impute_cat',\n",
       "                                                                   SimpleImputer(strategy='most_frequent')),\n",
       "                                                                  ('ohe',\n",
       "                                                                   OneHotEncoder(handle_unknown='ignore'))]),\n",
       "                                                  Index(['Gender', 'Vehicle_Age', 'Vehicle_Damage'], dtype='object')),\n",
       "                                                 ('num_encode',\n",
       "                                                  Pipeline(steps=[('impute_num',\n",
       "                                                                   SimpleImputer(strategy='median')),\n",
       "                                                                  ('scale',\n",
       "                                                                   StandardScaler())]),\n",
       "                                                  Index(['Age', 'Driving_License', 'Region_Code', 'Previously_Insured',\n",
       "       'Annual_Premium', 'Policy_Sales_Channel', 'Vintage'],\n",
       "      dtype='object'))])),\n",
       "                ('model', LogisticRegression())])"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_pipeline.fit(train_X, train_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "51b7b7e3-1a76-4122-99a5-5058dd143d51",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_pipeline.predict(train_X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "f84544a5-00ef-4c27-a6bd-b1aff47e6b13",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import f1_score\n",
    "def model_train_val_eval(train_X,val_X,train_y,val_y,model_pipeline):\n",
    "    pred_train = model_pipeline.predict(train_X)\n",
    "    pred_val = model_pipeline.predict(val_X)\n",
    "    \n",
    "    print('Train F1 Score:', f1_score(train_y, pred_train))\n",
    "    print('val F1 Score:', f1_score(val_y, pred_val))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "c3e9ebea-e3bd-4962-92fb-305912d3f8a9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train F1 Score: 0.00033376245649167983\n",
      "val F1 Score: 0.0\n"
     ]
    }
   ],
   "source": [
    "model_train_val_eval(train_X,val_X,train_y,val_y,model_pipeline)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "b92039c8-8001-4cf2-99ab-112ca0064524",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[9.99554679e-01, 4.45321250e-04],\n",
       "       [7.21305486e-01, 2.78694514e-01],\n",
       "       [6.86945814e-01, 3.13054186e-01],\n",
       "       ...,\n",
       "       [8.42730824e-01, 1.57269176e-01],\n",
       "       [8.09290934e-01, 1.90709066e-01],\n",
       "       [6.87480185e-01, 3.12519815e-01]])"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "model_pipeline.predict_proba(val_X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "12efc559-02ab-4bb5-a018-27e9f1b232ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "def model_train_val_eval(train_X,val_X,train_y,val_y,model_pipeline):\n",
    "    \n",
    "    predicted_train_tgt = model_pipeline.predict(train_X)\n",
    "    predicted_val_tgt = model_pipeline.predict(val_X)\n",
    "\n",
    "    print('Train AUC', roc_auc_score(train_y,predicted_train_tgt),sep='\\n')\n",
    "    print('Valid AUC', roc_auc_score(val_y,predicted_val_tgt),sep='\\n')\n",
    "\n",
    "    print('Train cnf_matrix', confusion_matrix(train_y,predicted_train_tgt),sep='\\n')\n",
    "    print('Valid cnf_matrix', confusion_matrix(val_y,predicted_val_tgt),sep='\\n')\n",
    "\n",
    "    print('Train cls_rep', classification_report(train_y,predicted_train_tgt),sep='\\n')\n",
    "    print('Valid cls rep', classification_report(val_y,predicted_val_tgt),sep='\\n')\n",
    "\n",
    "    y_pred_proba = model_pipeline.predict_proba(val_X)[:,1]\n",
    "    plt.figure()\n",
    "    fpr, tpr, thrsh = roc_curve(val_y,y_pred_proba)\n",
    "    #roc_auc = auc(fpr, tpr)\n",
    "    \n",
    "    plt.plot(fpr, tpr)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "39c370b1-b8f2-4325-b144-3e5ec516308d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train AUC\n",
      "0.5000635495727982\n",
      "Valid AUC\n",
      "0.49998499759961595\n",
      "Train cnf_matrix\n",
      "[[301059     12]\n",
      " [ 41920      7]]\n",
      "Valid cnf_matrix\n",
      "[[33327     1]\n",
      " [ 4783     0]]\n",
      "Train cls_rep\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.88      1.00      0.93    301071\n",
      "           1       0.37      0.00      0.00     41927\n",
      "\n",
      "    accuracy                           0.88    342998\n",
      "   macro avg       0.62      0.50      0.47    342998\n",
      "weighted avg       0.82      0.88      0.82    342998\n",
      "\n",
      "Valid cls rep\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.87      1.00      0.93     33328\n",
      "           1       0.00      0.00      0.00      4783\n",
      "\n",
      "    accuracy                           0.87     38111\n",
      "   macro avg       0.44      0.50      0.47     38111\n",
      "weighted avg       0.76      0.87      0.82     38111\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model_train_val_eval(train_X,val_X,train_y,val_y,model_pipeline)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "9a38ccae-2a48-4f80-a989-ffeb5907a7c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "params = [\n",
    "    {\n",
    "    'model': [LogisticRegression()],\n",
    "    'model__penalty':['l2',None],\n",
    "    'model__C':[0.5,3]\n",
    "    }    \n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "ad56251e-25c9-4b6c-a7e8-8f4adbfee613",
   "metadata": {},
   "outputs": [],
   "source": [
    "grid = GridSearchCV(estimator=model_pipeline, param_grid=params, \n",
    "                    cv=2, scoring='roc_auc')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "2a888102-2af0-4baa-b9a1-8b97e4bbf18c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=2,\n",
       "             estimator=Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                                        ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                                         Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                                          SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                                         (&#x27;ohe&#x27;,\n",
       "                                                                                          OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                                         Index([&#x27;Gender&#x27;, &#x27;Vehicle_Age&#x27;, &#x27;Vehicle_Damage&#x27;], dtype=&#x27;object&#x27;)),\n",
       "                                                                        (&#x27;num_encode&#x27;,\n",
       "                                                                         Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                                          SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                                                         (&#x27;scale&#x27;,\n",
       "                                                                                          StandardScaler())]),\n",
       "                                                                         Index([&#x27;Age&#x27;, &#x27;Driving_License&#x27;, &#x27;Region_Code&#x27;, &#x27;Previously_Insured&#x27;,\n",
       "       &#x27;Annual_Premium&#x27;, &#x27;Policy_Sales_Channel&#x27;, &#x27;Vintage&#x27;],\n",
       "      dtype=&#x27;object&#x27;))])),\n",
       "                                       (&#x27;model&#x27;, LogisticRegression())]),\n",
       "             param_grid=[{&#x27;model&#x27;: [LogisticRegression(C=0.5)],\n",
       "                          &#x27;model__C&#x27;: [0.5, 3],\n",
       "                          &#x27;model__penalty&#x27;: [&#x27;l2&#x27;, None]}],\n",
       "             scoring=&#x27;roc_auc&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-10\" type=\"checkbox\" ><label for=\"sk-estimator-id-10\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GridSearchCV</label><div class=\"sk-toggleable__content\"><pre>GridSearchCV(cv=2,\n",
       "             estimator=Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                                        ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                                         Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                                          SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                                         (&#x27;ohe&#x27;,\n",
       "                                                                                          OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                                         Index([&#x27;Gender&#x27;, &#x27;Vehicle_Age&#x27;, &#x27;Vehicle_Damage&#x27;], dtype=&#x27;object&#x27;)),\n",
       "                                                                        (&#x27;num_encode&#x27;,\n",
       "                                                                         Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                                          SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                                                         (&#x27;scale&#x27;,\n",
       "                                                                                          StandardScaler())]),\n",
       "                                                                         Index([&#x27;Age&#x27;, &#x27;Driving_License&#x27;, &#x27;Region_Code&#x27;, &#x27;Previously_Insured&#x27;,\n",
       "       &#x27;Annual_Premium&#x27;, &#x27;Policy_Sales_Channel&#x27;, &#x27;Vintage&#x27;],\n",
       "      dtype=&#x27;object&#x27;))])),\n",
       "                                       (&#x27;model&#x27;, LogisticRegression())]),\n",
       "             param_grid=[{&#x27;model&#x27;: [LogisticRegression(C=0.5)],\n",
       "                          &#x27;model__C&#x27;: [0.5, 3],\n",
       "                          &#x27;model__penalty&#x27;: [&#x27;l2&#x27;, None]}],\n",
       "             scoring=&#x27;roc_auc&#x27;)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-11\" type=\"checkbox\" ><label for=\"sk-estimator-id-11\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: Pipeline</label><div class=\"sk-toggleable__content\"><pre>Pipeline(steps=[(&#x27;preprocess&#x27;,\n",
       "                 ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                                  (&#x27;ohe&#x27;,\n",
       "                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                                  Index([&#x27;Gender&#x27;, &#x27;Vehicle_Age&#x27;, &#x27;Vehicle_Damage&#x27;], dtype=&#x27;object&#x27;)),\n",
       "                                                 (&#x27;num_encode&#x27;,\n",
       "                                                  Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                                   SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                                  (&#x27;scale&#x27;,\n",
       "                                                                   StandardScaler())]),\n",
       "                                                  Index([&#x27;Age&#x27;, &#x27;Driving_License&#x27;, &#x27;Region_Code&#x27;, &#x27;Previously_Insured&#x27;,\n",
       "       &#x27;Annual_Premium&#x27;, &#x27;Policy_Sales_Channel&#x27;, &#x27;Vintage&#x27;],\n",
       "      dtype=&#x27;object&#x27;))])),\n",
       "                (&#x27;model&#x27;, LogisticRegression())])</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-12\" type=\"checkbox\" ><label for=\"sk-estimator-id-12\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">preprocess: ColumnTransformer</label><div class=\"sk-toggleable__content\"><pre>ColumnTransformer(transformers=[(&#x27;cat_encode&#x27;,\n",
       "                                 Pipeline(steps=[(&#x27;impute_cat&#x27;,\n",
       "                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),\n",
       "                                                 (&#x27;ohe&#x27;,\n",
       "                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),\n",
       "                                 Index([&#x27;Gender&#x27;, &#x27;Vehicle_Age&#x27;, &#x27;Vehicle_Damage&#x27;], dtype=&#x27;object&#x27;)),\n",
       "                                (&#x27;num_encode&#x27;,\n",
       "                                 Pipeline(steps=[(&#x27;impute_num&#x27;,\n",
       "                                                  SimpleImputer(strategy=&#x27;median&#x27;)),\n",
       "                                                 (&#x27;scale&#x27;, StandardScaler())]),\n",
       "                                 Index([&#x27;Age&#x27;, &#x27;Driving_License&#x27;, &#x27;Region_Code&#x27;, &#x27;Previously_Insured&#x27;,\n",
       "       &#x27;Annual_Premium&#x27;, &#x27;Policy_Sales_Channel&#x27;, &#x27;Vintage&#x27;],\n",
       "      dtype=&#x27;object&#x27;))])</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-13\" type=\"checkbox\" ><label for=\"sk-estimator-id-13\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">cat_encode</label><div class=\"sk-toggleable__content\"><pre>Index([&#x27;Gender&#x27;, &#x27;Vehicle_Age&#x27;, &#x27;Vehicle_Damage&#x27;], dtype=&#x27;object&#x27;)</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-14\" type=\"checkbox\" ><label for=\"sk-estimator-id-14\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-15\" type=\"checkbox\" ><label for=\"sk-estimator-id-15\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">OneHotEncoder</label><div class=\"sk-toggleable__content\"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div></div></div></div></div></div></div></div><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-16\" type=\"checkbox\" ><label for=\"sk-estimator-id-16\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">num_encode</label><div class=\"sk-toggleable__content\"><pre>Index([&#x27;Age&#x27;, &#x27;Driving_License&#x27;, &#x27;Region_Code&#x27;, &#x27;Previously_Insured&#x27;,\n",
       "       &#x27;Annual_Premium&#x27;, &#x27;Policy_Sales_Channel&#x27;, &#x27;Vintage&#x27;],\n",
       "      dtype=&#x27;object&#x27;)</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-17\" type=\"checkbox\" ><label for=\"sk-estimator-id-17\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SimpleImputer</label><div class=\"sk-toggleable__content\"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-18\" type=\"checkbox\" ><label for=\"sk-estimator-id-18\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">StandardScaler</label><div class=\"sk-toggleable__content\"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div></div></div><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-19\" type=\"checkbox\" ><label for=\"sk-estimator-id-19\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "GridSearchCV(cv=2,\n",
       "             estimator=Pipeline(steps=[('preprocess',\n",
       "                                        ColumnTransformer(transformers=[('cat_encode',\n",
       "                                                                         Pipeline(steps=[('impute_cat',\n",
       "                                                                                          SimpleImputer(strategy='most_frequent')),\n",
       "                                                                                         ('ohe',\n",
       "                                                                                          OneHotEncoder(handle_unknown='ignore'))]),\n",
       "                                                                         Index(['Gender', 'Vehicle_Age', 'Vehicle_Damage'], dtype='object')),\n",
       "                                                                        ('num_encode',\n",
       "                                                                         Pipeline(steps=[('impute_num',\n",
       "                                                                                          SimpleImputer(strategy='median')),\n",
       "                                                                                         ('scale',\n",
       "                                                                                          StandardScaler())]),\n",
       "                                                                         Index(['Age', 'Driving_License', 'Region_Code', 'Previously_Insured',\n",
       "       'Annual_Premium', 'Policy_Sales_Channel', 'Vintage'],\n",
       "      dtype='object'))])),\n",
       "                                       ('model', LogisticRegression())]),\n",
       "             param_grid=[{'model': [LogisticRegression(C=0.5)],\n",
       "                          'model__C': [0.5, 3],\n",
       "                          'model__penalty': ['l2', None]}],\n",
       "             scoring='roc_auc')"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid.fit(train_X, train_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "f3b9e3b7-e777-47c0-be58-6cf22f73cd08",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'model': LogisticRegression(C=0.5), 'model__C': 0.5, 'model__penalty': 'l2'}"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grid.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "55e33b54-31be-4b03-8a4c-d9de728c0d50",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>params</th>\n",
       "      <th>mean_test_score</th>\n",
       "      <th>rank_test_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>{'model': LogisticRegression(C=0.5), 'model__C': 0.5, 'model__penalty': 'l2'}</td>\n",
       "      <td>0.836339</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>{'model': LogisticRegression(C=0.5), 'model__C': 0.5, 'model__penalty': None}</td>\n",
       "      <td>0.836337</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>{'model': LogisticRegression(C=0.5), 'model__C': 3, 'model__penalty': 'l2'}</td>\n",
       "      <td>0.836337</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>{'model': LogisticRegression(C=0.5), 'model__C': 3, 'model__penalty': None}</td>\n",
       "      <td>0.836337</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                                          params  \\\n",
       "0  {'model': LogisticRegression(C=0.5), 'model__C': 0.5, 'model__penalty': 'l2'}   \n",
       "1  {'model': LogisticRegression(C=0.5), 'model__C': 0.5, 'model__penalty': None}   \n",
       "2    {'model': LogisticRegression(C=0.5), 'model__C': 3, 'model__penalty': 'l2'}   \n",
       "3    {'model': LogisticRegression(C=0.5), 'model__C': 3, 'model__penalty': None}   \n",
       "\n",
       "   mean_test_score  rank_test_score  \n",
       "0         0.836339                1  \n",
       "1         0.836337                3  \n",
       "2         0.836337                2  \n",
       "3         0.836337                3  "
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "res_df = pd.DataFrame(grid.cv_results_,)\n",
    "pd.set_option('display.max_colwidth',100)\n",
    "res_df[['params','mean_test_score','rank_test_score']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "b2284309-1408-4d48-97f5-14e82d866f24",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>Response</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>381110</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>381111</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>381112</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       id  Response\n",
       "0  381110         0\n",
       "1  381111         0\n",
       "2  381112         0"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# read the submission file\n",
    "# predict with the last model\n",
    "#upload into the Analytic Vidya website\n",
    "\n",
    "sub = pd.read_csv('../Data/sample.csv')\n",
    "sub.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "6ca0e4d1-e449-4c40-a773-553d9eb0a0c8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Driving_License</th>\n",
       "      <th>Region_Code</th>\n",
       "      <th>Previously_Insured</th>\n",
       "      <th>Vehicle_Age</th>\n",
       "      <th>Vehicle_Damage</th>\n",
       "      <th>Annual_Premium</th>\n",
       "      <th>Policy_Sales_Channel</th>\n",
       "      <th>Vintage</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>381110</td>\n",
       "      <td>Male</td>\n",
       "      <td>25</td>\n",
       "      <td>1</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>&lt; 1 Year</td>\n",
       "      <td>No</td>\n",
       "      <td>35786.0</td>\n",
       "      <td>152.0</td>\n",
       "      <td>53</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>381111</td>\n",
       "      <td>Male</td>\n",
       "      <td>40</td>\n",
       "      <td>1</td>\n",
       "      <td>28.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1-2 Year</td>\n",
       "      <td>Yes</td>\n",
       "      <td>33762.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>111</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>381112</td>\n",
       "      <td>Male</td>\n",
       "      <td>47</td>\n",
       "      <td>1</td>\n",
       "      <td>28.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1-2 Year</td>\n",
       "      <td>Yes</td>\n",
       "      <td>40050.0</td>\n",
       "      <td>124.0</td>\n",
       "      <td>199</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       id Gender  Age  Driving_License  Region_Code  Previously_Insured  \\\n",
       "0  381110   Male   25                1         11.0                   1   \n",
       "1  381111   Male   40                1         28.0                   0   \n",
       "2  381112   Male   47                1         28.0                   0   \n",
       "\n",
       "  Vehicle_Age Vehicle_Damage  Annual_Premium  Policy_Sales_Channel  Vintage  \n",
       "0    < 1 Year             No         35786.0                 152.0       53  \n",
       "1    1-2 Year            Yes         33762.0                   7.0      111  \n",
       "2    1-2 Year            Yes         40050.0                 124.0      199  "
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "24331b66-1fa6-48f2-882e-cc3e10e18fd4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Response'], dtype='object')"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.columns.difference(test.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "2a4e42eb-d944-4b25-a306-fe62cb24a36c",
   "metadata": {},
   "outputs": [],
   "source": [
    "sub['Response'] = model_pipeline.predict(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "eb8629be-c0ca-4ab0-a140-4397d704314d",
   "metadata": {},
   "outputs": [],
   "source": [
    "sub.to_csv('../Data/submission.csv',index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "950b6e19-32dc-4736-9e38-d6a28654abbf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>Response</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>381110</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>381111</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>381112</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>381113</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>381114</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>127032</th>\n",
       "      <td>508142</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>127033</th>\n",
       "      <td>508143</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>127034</th>\n",
       "      <td>508144</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>127035</th>\n",
       "      <td>508145</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>127036</th>\n",
       "      <td>508146</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>127037 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "            id  Response\n",
       "0       381110         0\n",
       "1       381111         0\n",
       "2       381112         0\n",
       "3       381113         0\n",
       "4       381114         0\n",
       "...        ...       ...\n",
       "127032  508142         0\n",
       "127033  508143         0\n",
       "127034  508144         0\n",
       "127035  508145         0\n",
       "127036  508146         0\n",
       "\n",
       "[127037 rows x 2 columns]"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "c24372fb-6f4e-48d0-9ca7-88da9e17b9fd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['jobchg_pipeline_model.pkl']"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "joblib.dump(model_pipeline,'jobchg_pipeline_model.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "023d523a-ade5-42c5-8bce-ee7ec2fe061e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from imblearn.over_sampling import RandomOverSampler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "bd9dcd4f-cf89-4dc4-b2c2-b2a528d86227",
   "metadata": {},
   "outputs": [],
   "source": [
    "over_sampling = RandomOverSampler()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "3b66345a-7e3a-4641-8fb9-570d82a0fd1c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Response\n",
       "0           301071\n",
       "1            41927\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_y.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "89cc0ecd-bc7b-492e-b75f-f66aa0a84f15",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
