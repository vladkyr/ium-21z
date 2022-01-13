import pandas


class BaseModel:
    def __init__(self, train_data):
        self.avg_per_prod = train_data.groupby(by='product_id').mean().round()
        self.avg_per_prod['amount'] = self.avg_per_prod['amount'].astype(int)

    def predict(self, prod_id):
        return self.avg_per_prod.loc[prod_id, :][0]
