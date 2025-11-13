def annualize_daily_rate(daily_rate):
    return (1.0 + daily_rate)**252 - 1.0

