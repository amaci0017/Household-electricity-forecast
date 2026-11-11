import pandas as pd
import matplotlib.pyplot as plt

# pd.read_csv allows me to have access to the file
df = pd.read_csv("household_power_consumption.txt",
                 sep=";", na_values="?", low_memory=False)
#
df["time"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
s = df.set_index("time")["Global_active_power"].resample("h").mean()
s.head()

print(len(s)) # number of rows in total
print(s.isna().sum()) # numbers of rows with missing values
s["2007-01-01":"2007-01-07"].plot(figsize=(12, 4))
plt.show()

# start the estimation part
# baseline (simple guess)
data = pd.DataFrame({"actual": s,
                     "yesterday": s.shift(24),
                     "last_week": s.shift(168)}) # shift allows me to get down 
data.head()

data = data.dropna() # removes rows with missing values (Nan)
data.head()
test = data.loc["2010-06-01":] # loc picks row by their label

print(test["actual"].mean())
print(len(test), test.index[0])
miss = test["actual"] - test["yesterday"]
miss1 = test["actual"] - test["last_week"]
print(f"Same hour yesterday's consumption is off by about {miss.abs().mean()} kW on average ")
print(f"Same hour last week is off by about {miss1.abs().mean()} kW on average ")
# the best baseline (yesterday) is off by about 52% of the average power

# to make the model more acurate by adding features which are like clues
data["hour"] = data.index.hour
data["dayofweek"] = data.index.dayofweek
data["month"] = data.index.month
data.head()


feature = ["yesterday", "last_week", "hour", "dayofweek", "month"]
X = data[feature]
y = data["actual"]
print(X.head())
print(y.head())

# to train the model (training data)
X_train = X.loc[:"2010-05-31"]
y_train = y.loc[:"2010-05-31"]
# to test the model
X_test = X.loc["2010-06-01":]
y_test = y.loc["2010-06-01":]

print(len(X_train), len(y_train), len(X_test), len(y_test))

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test) # gives the train model only the clues/features
miss = y_test - y_pred
print(miss.abs().mean())


from sklearn.ensemble import GradientBoostingRegressor
model2 = GradientBoostingRegressor(random_state=0)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
miss2 = y_test - y_pred2
print(miss2.abs().mean())
# GradientBoostingRegressor is better

# now i compare the actual data to the predicted one (what the model predicted)
y_pred_s = pd.Series(y_pred2, index=y_test.index)
compare = pd.DataFrame({"actual": y_test, "predicted": y_pred_s})
ax = compare["2010-06-07":"2010-06-13"].plot(figsize=(12, 4))
ax.set_ylabel("Global_active_power (kW)")
ax.set_title("Model vs reality, 7 to 13 June 2010")
plt.show()
# the model failed to provide acurate predictions based on the graph, it misses peaks and dips...

compare["error"] = (compare["actual"] - compare["predicted"]).abs()
hourly_error = compare.groupby(compare.index.hour)["error"].mean()
ax = hourly_error.plot(kind="bar", figsize=(12, 4))
ax.set_title("Average error by hour of day")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Average error (kW)")


print("hours in test:", len(test))
print("average power:", test["actual"].mean())
print("baseline yesterday:", (test["actual"] - test["yesterday"]).abs().mean())
print("baseline last week:", (test["actual"] - test["last_week"]).abs().mean())


# see how the model predicts in general

pred_s = pd.Series(y_pred2, index=y_test.index)
both = pd.DataFrame({"actual": y_test,
                     "forecast": pred_s,
                     "baseline": X_test["yesterday"]})

daily = both.resample("D").sum()
daily["hours"] = both["actual"].resample("D").count()
daily =daily[daily["hours"]==24]
print(len(daily))

model_miss = (daily["actual"] - daily["forecast"]).abs().mean()
base_miss  = (daily["actual"] - daily["baseline"]).abs().mean()

print("average daily total:", daily["actual"].mean())
print("model miss per day:", model_miss)
print("baseline miss per day:", base_miss)

daily["model_miss"]= (daily["actual"] - daily["forecast"]).abs()
daily["base_miss"]= (daily["actual"] - daily["baseline"]).abs()
print((daily["model_miss"] < daily["base_miss"]).mean())

# making a forecast function (testing for a day)
def forecast_day(day):
    if day < "2010-06-01":
        print("The model was trained on this day so pick 2010-06-01 or later")
    try:
        X_day = X.loc[day] 
    except KeyError:
        print("There's no data for that date") 
        return

    if len(X_day)<24:
        print("This date has missing hours, try another")
        return
    forecast = pd.Series(model2.predict(X_day), index=X_day.index)
    actual = y.loc[day]
    ax = pd.DataFrame ({"actual":actual, "forecast":forecast}).plot(figsize=(10,4))
    ax.set_ylabel("Power (kW)")
    ax.set_title("Forecast for " + day)
    ax.figure.savefig("forecast_" + day + ".png", dpi=150, bbox_inches="tight")
    print("forecast total: ", round(forecast.sum(),1),"kWh")
    print("actual total: " , round(actual.sum(),1),"kWh")


forecast_day("2010-07-15")



