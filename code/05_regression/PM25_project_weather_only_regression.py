import pandas as pd
import statsmodels.api as sm

# 读取数据
df = pd.read_csv("step5_multiple_regression.csv")

# 日期处理
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# 因变量：PM2.5
y = df["PM2.5"]

# 自变量
X = df[
    [
        "Temperature",
        "Humidity",
        "WindSpeed",
        "Precipitation",
        "Lag_PM2.5",
        "season_Spring",
        "season_Summer",
        "season_Autumn"
    ]
]

# 添加截距项
X = sm.add_constant(X)

# 建立线性回归模型
model = sm.OLS(y, X).fit()

# 输出回归结果
print(model.summary())

def predict_pm25(temperature, humidity, wind_speed, precipitation, lag_pm25, season):
    """
    根据多元线性回归结果预测 PM2.5 浓度
    
    参数：
    temperature: 温度
    humidity: 湿度
    wind_speed: 风速
    precipitation: 降水量
    lag_pm25: 前一天 PM2.5 浓度
    season: 季节，可输入 "Spring", "Summer", "Autumn", "Winter"
    """

    # 截距
    const = 11.6528

    # 回归系数
    coef_temperature = 0.0345
    coef_humidity = 0.3260
    coef_wind_speed = -5.9320
    coef_precipitation = -0.4154
    coef_lag_pm25 = 0.5398

    coef_spring = 5.4511
    coef_summer = -10.9597
    coef_autumn = -6.8666

    # 季节虚拟变量
    season_spring = 1 if season == "Spring" else 0
    season_summer = 1 if season == "Summer" else 0
    season_autumn = 1 if season == "Autumn" else 0

    # 预测公式
    predicted_pm25 = (
        const
        + coef_temperature * temperature
        + coef_humidity * humidity
        + coef_wind_speed * wind_speed
        + coef_precipitation * precipitation
        + coef_lag_pm25 * lag_pm25
        + coef_spring * season_spring
        + coef_summer * season_summer
        + coef_autumn * season_autumn
    )

    return predicted_pm25

print("\n现在可以输入气象数据来预测 PM2.5")
print("季节请输入：春季、夏季、秋季、冬季")

# 1. 引导用户输入各项数值
temperature = float(input("请输入温度 Temperature："))
humidity = float(input("请输入湿度 Humidity："))
wind_speed = float(input("请输入风速 WindSpeed："))
precipitation = float(input("请输入降水量 Precipitation："))
lag_pm25 = float(input("请输入前一天 PM2.5 Lag_PM2.5："))
season = input("请输入季节：")

# 2. 根据输入的季节设置虚拟变量
season_spring = 0
season_summer = 0
season_autumn = 0

if season == "春季" or season == "spring":
    season_spring = 1
elif season == "夏季" or season == "summer":
    season_summer = 1
elif season == "秋季" or season == "autumn":
    season_autumn = 1
elif season == "冬季" or season == "winter":
    pass
else:
    print("季节输入错误，请输入：春季、夏季、秋季、冬季")
    exit()

# 3. 构造新的预测数据
new_data = pd.DataFrame({
    "const": [1],
    "Temperature": [temperature],
    "Humidity": [humidity],
    "WindSpeed": [wind_speed],
    "Precipitation": [precipitation],
    "Lag_PM2.5": [lag_pm25],
    "season_Spring": [season_spring],
    "season_Summer": [season_summer],
    "season_Autumn": [season_autumn]
})

# 4. 使用已经训练好的模型预测
predicted_pm25 = model.predict(new_data)[0]

# 5. 输出预测结果
print("\n预测结果：")
print(f"预测 PM2.5 浓度为：{predicted_pm25:.2f}")