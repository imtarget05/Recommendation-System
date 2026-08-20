from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402


@pytest.fixture()
def sample_movielens_data(tmp_path):  # noqa: ANN001, ANN202
    """Tiny MovieLens-like source files for offline normalization tests."""
    ratings_csv = tmp_path / "ratings.csv"
    ratings_csv.write_text(
        "userId,movieId,rating,timestamp\n"
        "1,10,4.5,1000000000\n"
        "1,11,2.0,1000000001\n"
        "2,10,5.0,1000000002\n"
        "3,12,3.0,1000000003\n"
        "1,13,4.0,1000000004\n"
    )
    movies_csv = tmp_path / "movies.csv"
    movies_csv.write_text(
        "movieId,title,genres\n"
        "10,Toy Story (1995),Adventure|Animation|Children\n"
        "11,Ghost (1990),Comedy|Romance\n"
        "12,Jumanji (1995),Adventure|Children|Fantasy\n"
        "13,Alien (1979),Horror|Sci-Fi\n"
    )
    tags_csv = tmp_path / "tags.csv"
    tags_csv.write_text(
        "userId,movieId,tag,timestamp\n"
        "1,10,pixar,1000000000\n"
        "1,10,animated,1000000000\n"
        "2,12,board game,1000000000\n"
    )
    return {"ratings": ratings_csv, "movies": movies_csv, "tags": tags_csv}


@pytest.fixture()
def sample_hm_data(tmp_path):  # noqa: ANN001, ANN202
    """Tiny H&M-like source files (Kaggle layout) for offline normalization tests."""
    tx_csv = tmp_path / "transactions_train.csv"
    tx_csv.write_text(
        "t_dat,customer_id,article_id,price,sales_channel_id\n"
        "2018-09-20,12345,0000010001,19.99,2\n"
        "2018-09-21,12345,0000010002,29.99,1\n"
        "2018-09-22,202,0000010003,39.99,2\n"
    )
    articles_csv = tmp_path / "articles.csv"
    articles_csv.write_text(
        "article_id,prod_name,product_type_name,product_group_name,"
        "graphical_appearance_name,colour_group_name,index_group_name,"
        "section_name,detail_desc\n"
        "0000010001,Black Slim Fit Jacket,Jacket,Apparel,Solid,Black,"
        "Menswear,Outerwear,A stylish black jacket.\n"
        "0000010002,White T Shirt,T-Shirt Apparel,Apparel,Print,White,"
        "Ladieswear,Tops,\n"
        "0000010003,Running Shoes,Shoes,Footwear,Plain,Grey,"
        "Menswear,Footwear,Light running shoes.\n"
    )
    customers_csv = tmp_path / "customers.csv"
    customers_csv.write_text(
        "customer_id,postal_code,club_member_status\n"
        "12345,50200,ACTIVE\n"
        "202,10115,ACTIVE\n"
    )
    return {"transactions": tx_csv, "articles": articles_csv, "customers": customers_csv}
