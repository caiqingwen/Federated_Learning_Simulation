import models
import torch


# 客户端类
class Client(object):
    # 构造函数
    def __init__(self, conf, model, train_dataset, id=-1):
        # 读取配置文件
        self.conf = conf
        # 根据配置文件获取客户端本地模型（一般由服务器传输）
        self.local_model = models.get_model(self.conf["model_name"])
        # 客户端ID
        self.client_id = id
        # 客户端本地数据集
        self.train_dataset = train_dataset
        # 按ID对数据集集合进行拆分
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        train_indices = all_range[id * data_len: (id + 1) * data_len]
        # 生成数据加载器
        self.train_loader = torch.utils.data.DataLoader(
            # 指定父集合
            self.train_dataset,
            # 每个batch加载多少样本
            batch_size=conf["batch_size"],
            # 指定子集合
            # sampler定义从数据集中提取样本的策略
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        )

    # 模型本地训练函数
    def local_train(self, model):
        # 客户端获取服务器的模型，然后通过部分本地数据集进行训练
        for name, param in model.state_dict().items():
            # 用服务器下发的全局模型覆盖本地模型
            self.local_model.state_dict()[name].copy_(param.clone())
        # 定义最优化函数器用户本地模型训练
        optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.conf['lr'],
            momentum=self.conf['momentum']
        )
        # 本地训练模型
        # 设置开启模型训练
        self.local_model.train()
        # 开始训练模型
        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                # 如果可以的话加载到gpu
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                # 梯度初始化为0
                optimizer.zero_grad()
                # 训练预测
                output = self.local_model(data)
                # 计算损失函数cross_entropy交叉熵误差
                loss = torch.nn.functional.cross_entropy(output, target)
                # 反向传播
                loss.backward()
                # 更新参数
                optimizer.step()
            print("Epoch %d done." % e)
        # 创建差值字典（结构与模型参数同规格），用于记录差值
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            # 计算训练后与训练前的差值
            diff[name] = (data - model.state_dict()[name])
            print("Client %d local train done" % self.client_id)
        # 客户端返回差值
        return diff