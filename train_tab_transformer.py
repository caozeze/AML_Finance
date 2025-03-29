import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.tab_transformer import TabTransformer

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=10):
    best_val_loss = float('inf')
    no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

    return best_model_state

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    train_data = pd.read_csv('./data/clean_train.csv')
    # test_data = pd.read_csv('./data/clean_test.csv')
    
    # 准备特征和标签
    features = train_data.drop(['credit_score','Unnamed: 0'], axis=1)
    labels = train_data['credit_score']

    for column in features.columns:
        # 检查是否为对象类型
        if features[column].dtype == 'object':
            # 尝试转换为数值类型
            features[column] = pd.to_numeric(features[column], errors='coerce')
        # 检查是否为布尔类型
        elif features[column].dtype == 'bool':
            features[column] = features[column].astype(int)
    
    # 转换为PyTorch张量
    X = torch.FloatTensor(features.values)
    y = torch.LongTensor(labels.values)
    
    # 设置交叉验证
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # 模型参数
    input_dim = features.shape[1]
    num_classes = len(labels.unique())
    
    # 存储每个fold的结果
    fold_results = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X)):
        print(f'\nFold {fold + 1}/{k_folds}')
        
        # 准备数据加载器
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X[train_ids], y[train_ids]),
            batch_size=128, num_workers=8, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X[val_ids], y[val_ids]),
            batch_size=128, num_workers=8, 
        )
        
        # 初始化模型
        model = TabTransformer(input_dim=input_dim, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型
        best_model_state = train_model(model, train_loader, val_loader, criterion, optimizer, device)
        
        # 加载最佳模型状态
        model.load_state_dict(best_model_state)
        
        # 评估模型
        fold_result = evaluate_model(model, val_loader, device)
        fold_results.append(fold_result)
        print(f'Fold {fold + 1} Results:')
        for metric, value in fold_result.items():
            print(f'{metric}: {value:.4f}')

        # 保存模型和评估结果
        model_save_path = f'./models/tab_transformer_fold_{fold+1}_f1_{fold_result["f1_score"]:.4f}.pth'
        torch.save({
            'model_state_dict': best_model_state,
            'model_config': {
                'input_dim': input_dim,
                'num_classes': num_classes
            },
            'metrics': fold_result,
            'fold': fold + 1
        }, model_save_path)
        print(f'模型已保存到: {model_save_path}')
    
    # 计算平均结果
    avg_results = {}
    for metric in fold_results[0].keys():
        avg_results[metric] = np.mean([result[metric] for result in fold_results])
    
    print('\nAverage Cross-validation Results:')
    for metric, value in avg_results.items():
        print(f'{metric}: {value:.4f}')

if __name__ == '__main__':
    main()