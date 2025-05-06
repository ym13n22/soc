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

    #print("user_id\titem_id\trating\tpredicted_rating\tabs_error")
    for user_id, item_id, true_rating in rows:
        try:
            pred_rating = model.predict(int(user_id), int(item_id))
            abs_error = abs(true_rating - pred_rating)
            total_error += abs_error
            valid_count += 1
            #print(f"{user_id}\t{item_id}\t{true_rating:.2f}\t{pred_rating:.2f}\t\t{abs_error:.4f}")
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

    print("开始自动调参...")
    best_model, best_config = MatrixFactorization.grid_search_train(
        ratings=ratings,
        val_ratings=val_ratings,
        num_users=num_users,
        num_items=num_items,
        k_list=[250,300,350],
        lr_list=[0.01,0.011,0.012,0.013,0.014,0.015],
        reg_list=[0.03,0.035,0.04,0.045, 0.05],
        epochs=10
    )
    print(f"最优参数：k={best_config['k']}, lr={best_config['lr']}, reg={best_config['reg']}")

    # 用最优模型预测测试集
    print("使用最优模型在 testTableSplit 上进行预测并评估...")
    predict_localtest_sqlite(best_model, db_path)


    # 初始化模型
    #model = MatrixFactorization(num_users, num_items, k=250, lr=0.015, reg=0.035, epochs=50 )
    #0.6237
    #50 0.5981
    #k=50 0.5971
    #k=100 0.5958 0.6207(10)
    #k=150 0.5955  0.6188(10)
    #print("initialize model")
    # 训练模型
    #model.train(ratings,val_ratings)
    #print("trainModel")

    #predict_localtest_sqlite(model, db_path)




# 2. 矩阵分解实现（使用 SGD）
class MatrixFactorization:
    def __init__(self, num_users, num_items, k=20, lr=0.01, reg=0.1, epochs=5):
        self.k = k
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

        self.U = np.random.normal(0, 0.1, (num_users, k))
        self.V = np.random.normal(0, 0.1, (num_items, k))

        self.b_u = np.zeros(num_users)
        self.b_i = np.zeros(num_items)
        self.mu = 0

    def train(self, ratings, val_ratings=None, verbose=True, early_stop_rounds=5):
        self.mu = np.mean([r for _, _, r in ratings])
        best_mae = float('inf')
        no_improve_count = 0

        for epoch in range(self.epochs):
            np.random.shuffle(ratings)
            for user, item, rating in ratings:
                user = int(user)
                item = int(item)

                p_u = self.U[user]
                q_i = self.V[item]
                b_u = self.b_u[user]
                b_i = self.b_i[item]

                pred = self.mu + b_u + b_i + np.dot(p_u, q_i)
                e_ui = rating - pred

                self.b_u[user] += self.lr * (e_ui - self.reg * b_u)
                self.b_i[item] += self.lr * (e_ui - self.reg * b_i)
                self.U[user] += self.lr * (e_ui * q_i - self.reg * p_u)
                self.V[item] += self.lr * (e_ui * p_u - self.reg * q_i)

            if val_ratings is not None:
                val_mae = self.evaluate(val_ratings)
                if verbose:
                    print(f"Epoch {epoch + 1}/{self.epochs} completed. Validation MAE: {val_mae:.4f}")

                # Early stopping logic
                if val_mae + 1e-4 < best_mae:  # 加一点容差防止浮动误判
                    best_mae = val_mae
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= early_stop_rounds:
                        if verbose:
                            print(f"⏹️ Early stopping at epoch {epoch + 1} with best MAE {best_mae:.4f}")
                        break
            elif verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} completed.")

    def predict(self, user, item):
        return self.mu + self.b_u[user] + self.b_i[item] + np.dot(self.U[user], self.V[item])

    def evaluate(self, val_ratings):
        errors = []
        for user, item, true_rating in val_ratings:
            try:
                pred_rating = self.predict(int(user), int(item))
                errors.append(abs(true_rating - pred_rating))
            except IndexError:
                continue
        return np.mean(errors) if errors else float('nan')

    @staticmethod
    def grid_search_train(ratings, val_ratings, num_users, num_items,
                          k_list=[20], lr_list=[0.01], reg_list=[0.1], epochs=5):
        best_model = None
        best_config = None
        best_mae = float('inf')
        results = []

        for k in k_list:
            for lr in lr_list:
                for reg in reg_list:
                    print(f"Training with k={k}, lr={lr}, reg={reg} ...")
                    model = MatrixFactorization(num_users, num_items, k=k, lr=lr, reg=reg, epochs=epochs)
                    model.train(ratings, val_ratings, verbose=False, early_stop_rounds=3)
                    mae = model.evaluate(val_ratings)
                    print(f"→ Validation MAE = {mae:.4f}\n")

                    results.append((k, lr, reg, mae))

                    if mae < best_mae:
                        best_mae = mae
                        best_model = model
                        best_config = {'k': k, 'lr': lr, 'reg': reg}

        print("\n📊 All results:")
        for k, lr, reg, mae in results:
            print(f"  k={k:<4} | lr={lr:<6} | reg={reg:<5} => MAE = {mae:.4f}")

        print("\n✅ Best configuration:")
        print(f"  k={best_config['k']}, lr={best_config['lr']}, reg={best_config['reg']}")
        print(f"  Best Validation MAE = {best_mae:.4f}")

        return best_model, best_config





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