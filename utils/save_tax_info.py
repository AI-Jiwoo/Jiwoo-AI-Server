from utils.save_IncomRates import SaveIncomeRate
from utils.save_VATInfo import SaveVATInfo
from utils.save_simpleTransaction import SaveSimpleTransaction
from utils.save_tax_calcution_flow import SaveTaxCalculationFlow
from utils.save_tax_method_url import SaveTaxMethodUrl


save_income_rate = SaveIncomeRate()
save_VAT_info = SaveVATInfo()
save_simple_transaction = SaveSimpleTransaction()
save_tax_calculation_flow = SaveTaxCalculationFlow()
save_tax_method_url = SaveTaxMethodUrl()

class SaveTaxInfo : 
    def save_tax_info (self, category : str):
        if category == 'IncomeRates':
            save_income_rate.save_incomeRates()

        elif category == 'VATInfo':
            save_VAT_info.save_VAT_info()
            
        elif category == 'SimpleTransaction':
            save_simple_transaction.save_simple_transaction()

        elif category == 'TaxFlow' :
            save_tax_calculation_flow.save_tax_flow()

        elif category == 'TaxMethod' :
            save_tax_method_url.save_tax_method_url()

        else : 
            raise ValueError(f"알 수 없는 카테고리 입니다. {category}")