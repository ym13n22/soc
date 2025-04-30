import sqlite3
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def create_db(csvfile_path,db_path):
    df = pd.read_csv(csvfile_path,header=None, names=["user_id", "item_id","rating", "timestamp"])  # 替换为你的文件路径

    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor=conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trainTable (
            user_id INTEGER,
            item_id INTEGER,
            rating REAL,
            timestamp DATETIME
        )
    ''')

    cursor.execute('''
            CREATE TABLE IF NOT EXISTS trainTableSplit (
                user_id INTEGER,
                item_id INTEGER,
                rating REAL,
                timestamp DATETIME
            )
        ''')

    cursor.execute('''
            CREATE TABLE IF NOT EXISTS testTableSplit (
                user_id INTEGER,
                item_id INTEGER,
                rating REAL,
                timestamp DATETIME
            )
        ''')

    for row in df.itertuples(index=False):
        cursor.execute("INSERT INTO trainTable (user_id, item_id,rating,timestamp) VALUES (?, ?, ?, ?)",
                       (row.user_id, row.item_id,row.rating, row.timestamp))

    for row in train_data.itertuples(index=False):
        cursor.execute("INSERT INTO trainTableSplit (user_id, item_id,rating,timestamp) VALUES (?, ?, ?, ?)",
                       (row.user_id, row.item_id,row.rating, row.timestamp))

    for row in test_data.itertuples(index=False):
        cursor.execute("INSERT INTO testTableSplit (user_id, item_id,rating,timestamp) VALUES (?, ?, ?, ?)",
                       (row.user_id, row.item_id,row.rating, row.timestamp))

    conn.commit()
    conn.close()



# 1. 加载评分数据（从 SQLite 数据库）
def load_ratings(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(user_id), MAX(item_id) FROM trainTable")
    max_user, max_item = cursor.fetchone()

    cursor.execute("SELECT user_id, item_id, rating FROM trainTable")
    data = cursor.fetchall()
    conn.close()
    data = np.array(data)
    train_ratings, val_ratings = train_test_split(data, test_size=0.2, random_state=42)
    return train_ratings,val_ratings, max_user + 1, max_item + 1


def load_localTest_ratings(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(user_id), MAX(item_id) FROM trainTableSplit")
    max_user, max_item = cursor.fetchone()

    cursor.execute("SELECT user_id, item_id, rating FROM trainTableSplit")
    data = cursor.fetchall()
    conn.close()
    data = np.array(data)
    train_ratings, val_ratings = train_test_split(data, test_size=0.2, random_state=42)
    return train_ratings,val_ratings, max_user + 1, max_item + 1


def predict_test_file(model, test_file_path, output_file_path):
    # 加载测试数据（CSV），格式为: user_id,item_id,timestamp
    df = pd.read_csv(test_file_path)

    # 预测评分
    predictions = []
    for _, row in df.iterrows():
        user = int(row['user_id'])
        item = int(row['item_id'])
        # 若模型中不存在该用户或物品，可以跳过或填默认值
        try:
            rating = model.predict(user, item)
        except IndexError:
            rating = np.nan  # 或设为模型平均值等
        predictions.append(rating)

    # 插入预测列
    df.insert(loc=2, column='predicted_rating', value=predictions)

    # 保存到新文件
    df.to_csv(output_file_path, index=False)
    print(f"预测结果已保存到: {output_file_path}")

def predict_localtest_sqlite(model, db_path):
    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 从表中读取数据
    query = f"SELECT user_id, item_id, rating FROM testTableSplit"
    cursor.execute(query)
    rows = cursor.fetchall()

    total_error = 0
    valid_count = 0

    print("user_id\titem_id\trating\tpredicted_rating\tabs_error")
    for user_id, item_id, true_rating in rows:
        try:
            pred_rating = model.predict(int(user_id), int(item_id))
            abs_error = abs(true_rating - pred_rating)
            total_error += abs_error
            valid_count += 1
            print(f"{user_id}\t{item_id}\t{true_rating:.2f}\t{pred_rating:.2f}\t\t{abs_error:.4f}")
        except IndexError:
            # 模型中没有该用户或物品的向量，跳过或处理
            continue

    # 计算 MAE
    if valid_count > 0:
        mae = total_error / valid_count
        print(f"\nMean Absolute Error (MAE): {mae:.4f}")
    else:
        print("无有效预测样本，无法计算 MAE。")

    # 关闭数据库连接
    conn.close()


def local_test_file(db_path):
    ratings,val_ratings, num_users, num_items = load_localTest_ratings(db_path)

    # 初始化模型
    model = MatrixFactorization(num_users, num_items, k=30, lr=0.01, reg=0.05, epochs=10)
    print("initialize model")
    # 训练模型
    model.train(ratings,val_ratings)
    print("trainModel")

    predict_localtest_sqlite(model, db_path)




# 2. 矩阵分解实现（使用 SGD）
class MatrixFactorization:
    def __init__(self, num_users, num_items, k=20, lr=0.01, reg=0.1, epochs=5):
        self.k = k  # Latent factors
        self.lr = lr  # Learning rate (γ)
        self.reg = reg  # Regularization (λ)
        self.epochs = epochs  # Number of epochs
        self.U = np.random.normal(0, 0.1, (num_users, k))  # User latent matrix P
        self.V = np.random.normal(0, 0.1, (num_items, k))  # Item latent matrix Q

    def train(self, ratings,val_ratings=None):
        for epoch in range(self.epochs):
            np.random.shuffle(ratings)
            for user, item, rating in ratings:
                user = int(user)
                item = int(item)

                # Get current vectors
                p_u = self.U[user]
                q_i = self.V[item]

                # Predict and compute error
                pred = np.dot(p_u, q_i)
                e_ui = rating - pred

                # Update using Option 2 SGD formulas
                self.V[item] = q_i + self.lr * (e_ui * p_u - self.reg * q_i)
                self.U[user] = p_u + self.lr * (e_ui * self.V[item] - self.reg * p_u)

                # 验证集评估
            if val_ratings is not None:
                    val_mae = self.evaluate(val_ratings)
                    print(f"Epoch {epoch + 1}/{self.epochs} completed. Validation MAE: {val_mae:.4f}")
            else:
                    print(f"Epoch {epoch + 1}/{self.epochs} completed.")

    def evaluate(self, val_ratings):
        errors = []
        for user, item, true_rating in val_ratings:
            try:
                pred_rating = self.predict(int(user), int(item))
                errors.append(abs(true_rating - pred_rating))
            except IndexError:
                continue  # 用户或物品超出训练范围
        return np.mean(errors) if errors else float('nan')

    def predict(self, user, item):
        return np.dot(self.U[user], self.V[item])





# 3. 主程序入口
if __name__ == "__main__":
    # 加载数据
    db_path = "data/20M.db"
    csvfile_path = "train_20M_withratings.csv"
    test_file_path="test_20M_withoutratings.csv"
    output_file_path="test_20M_withratings.csv"
    #create_db(csvfile_path, db_path)
    local_test_file(db_path)

    
    '''
    ratings, num_users, num_items = load_ratings(db_path)

    # 初始化模型
    model = MatrixFactorization(num_users, num_items, k=30, lr=0.01, reg=0.05, epochs=10)

    # 训练模型
    model.train(ratings)

    predict_test_file(model, csvfile_path, db_path)
'''