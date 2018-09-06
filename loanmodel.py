
# coding: utf-8

# # Loan class defition

# In[1]:

import numpy as np
import pandas as pd
from datetime import date
from scipy import optimize


# In[2]:

# import datetime
# from scipy import optimize 

def secant_method(tol, f, x0):
    """
    Solve for x where f(x)=0, given starting x0 and tolerance.
    
    Arguments
    ----------
    tol: tolerance as percentage of final result. If two subsequent x values are with tol percent, the function will return.
    f: a function of a single variable
    x0: a starting value of x to begin the solver
    Notes
    ------
    The secant method for finding the zero value of a function uses the following formula to find subsequent values of x. 
    
    x(n+1) = x(n) - f(x(n))*(x(n)-x(n-1))/(f(x(n))-f(x(n-1)))
    
    Warning 
    --------
    This implementation is simple and does not handle cases where there is no solution. Users requiring a more robust version should use scipy package optimize.newton.
    """

    x1 = x0*1.1
    while (abs(x1-x0)/abs(x1) > tol):
        x0, x1 = x1, x1-f(x1)*(x1-x0)/(f(x1)-f(x0))
    return x1

def xnpv(rate,cashflows):
    """
    Calculate the net present value of a series of cashflows at irregular intervals.
    Arguments
    ---------
    * rate: the discount rate to be applied to the cash flows
    * cashflows: a list object in which each element is a tuple of the form (date, amount), where date is a python datetime.date object and amount is an integer or floating point number. Cash outflows (investments) are represented with negative amounts, and cash inflows (returns) are positive amounts.
    
    Returns
    -------
    * returns a single value which is the NPV of the given cash flows.
    Notes
    ---------------
    * The Net Present Value is the sum of each of cash flows discounted back to the date of the first cash flow. The discounted value of a given cash flow is A/(1+r)**(t-t0), where A is the amount, r is the discout rate, and (t-t0) is the time in years from the date of the first cash flow in the series (t0) to the date of the cash flow being added to the sum (t).  
    * This function is equivalent to the Microsoft Excel function of the same name. 
    """

    chron_order = sorted(cashflows, key = lambda x: x[0])
    t0 = chron_order[0][0] #t0 is the date of the first cash flow

    return sum([cf/(1+rate)**((t-t0).days/365.0) for (t,cf) in chron_order])

def xirr(cashflows,guess=0.1):
    """
    Calculate the Internal Rate of Return of a series of cashflows at irregular intervals.
    Arguments
    ---------
    * cashflows: a list object in which each element is a tuple of the form (date, amount), where date is a python datetime.date object and amount is an integer or floating point number. Cash outflows (investments) are represented with negative amounts, and cash inflows (returns) are positive amounts.
    * guess (optional, default = 0.1): a guess at the solution to be used as a starting point for the numerical solution. 
    Returns
    --------
    * Returns the IRR as a single value
    
    Notes
    ----------------
    * The Internal Rate of Return (IRR) is the discount rate at which the Net Present Value (NPV) of a series of cash flows is equal to zero. The NPV of the series of cash flows is determined using the xnpv function in this module. The discount rate at which NPV equals zero is found using the secant method of numerical solution. 
    * This function is equivalent to the Microsoft Excel function of the same name.
    * For users that do not have the scipy module installed, there is an alternate version (commented out) that uses the secant_method function defined in the module rather than the scipy.optimize module's numerical solver. Both use the same method of calculation so there should be no difference in performance, but the secant_method function does not fail gracefully in cases where there is no solution, so the scipy.optimize.newton version is preferred.
    """
    
    #return secant_method(0.0001,lambda r: xnpv(r,cashflows),guess)
    return optimize.newton(lambda r: xnpv(r,cashflows),guess)


# In[3]:

def calc_xnpv(df, rate):
    cashflow = [(df.iloc[idx, 0], df.iloc[idx, 1]) for idx in range(len(df))]
    return (xnpv(rate, cashflow))


# In[4]:

def calc_xirr(df, guess = 0.1):
    # it is assumed df has 2 columns only and column 0 dates and column 1 cashflow itself
    cashflows = [(df.iloc[idx, 0], df.iloc[idx, 1]) for idx in range(len(df))]
    return (xirr(cashflows, guess))


# In[5]:

def make_bullet_loan(start_date, years, frequency, par_value, investment_size, coupon, colnames):
    
    """
    We assume that notional amount stays constant throught the life of the loan, but purchase can be made under par or over par
    
    start_date : string expression of any date
    years : [x] years
    frequency : how many payments a year, ex 4 if quarterly
    investment_size = [1000000]
    par : investment_size may be larger or smaller than the par
    coupon : as % of the investment size
    colnames : define column names for ['date', days', 'notional','coupon','income', 'cash flow of principal' ,'net CF']
    """
    frequency_table = {1:'Y',
                      2:'6M',
                      4:'3M'}
    
    col_for_date = colnames[0]
    col_for_days = colnames[1]
    col_for_notional = colnames[2]
    col_for_principal = colnames[3] # this will be the basis of the manager's mgnt fees
    col_for_coupon = colnames[4]
    col_for_income = colnames[5]
    col_for_cf_principal = colnames[6] # accounts for deployment and recoupment of principals
    col_for_netCF = colnames[7] # sum of col_for_par and col_for_income
    
    rng = pd.date_range(start = pd.to_datetime(start_date), periods = frequency * years +1, # needs to 1 period to make it
                        freq = frequency_table[frequency])                                  # exactly designated years
    df = pd.DataFrame(index = rng, columns = colnames[1:], 
                      dtype = 'float')
    
    df.reset_index(inplace = True)
    
    df.rename(columns = {'index' : col_for_date}, inplace = True)
    
    # notional
    df[col_for_notional] = par_value
    
    # principal
    df[col_for_principal] = investment_size
        
    # calculating incomes
    df[col_for_coupon] = coupon
    df[col_for_days] = df[col_for_date].diff().dt.days # note that you need to convert df['date'].diff() which is a series
                                                       # that needs to be converted into numeric 
    df.loc[0,col_for_days] = 0 # replace NaN with 0
    df.loc[:,col_for_income] = df[col_for_notional] * df[col_for_coupon] * df[col_for_days] / 365

    # calculat cash flow related to principals
    df[col_for_cf_principal] = 0 # first designate 0 for all the rows
    df.loc[0, col_for_cf_principal] = - investment_size # first there is cash outflow at day 0
    df.loc[len(df)-1, col_for_cf_principal] = par_value # then we recoup at par at the end
    
    # calculate net cash flow for each period
    df[col_for_netCF] = df[col_for_income] + df[col_for_cf_principal]
    
    return df


# In[6]:

def convert_NaN(df, first_day, last_day, cf_item):
    """
    helper function to convert NaN to appropriate numbers when dataframe of 2 separate  loans are merged
    df : 2 column dataframe with column 0 has date and column 2 has data
    first_day, last_day : first date and last date of the loans - these are necessary as loan balance is 0 before first day
    ans after last day of the loan while it stays the same inbetween
    """
    data = list(df.iloc[:,1])
    dates = list(df.iloc[:,0])
    
    for idx in range(len(df)):
        if np.isnan(data[idx]):
            if dates[idx] <= first_day or dates[idx] >= last_day or cf_item:
                data[idx] = 0
            else:
                data[idx] = data[idx-1]
    
    df.iloc[:,1] = data
    
    return df


# In[7]:

def merge_cf(loan1, loan2, columns):
    """
    columns : {'date': False, 
                    'notional' : False, 
                    'invested' : False,
                    'income' : True,
                    'cash flow of principals' : True
                    'net CF' : True} would be merged on 'date'
    'date' must be the first one
    
    return summation of chosen columns from two loans
    
    """
    column_length = len(columns) # the merged dataframe would have (column_lengths * 2 + 1) columns
    column_keys = list(columns.keys())
    
    date_column_name = column_keys[0]
    
    # note the first day and last day of each loan - this is to be used for cleaning up NaN later
    l1 = loan1[column_keys].reset_index().iloc[:,1:]
    l2 = loan2.loc[:,column_keys].reset_index().iloc[:,1:]
    
    F_L_days = [(l1.iloc[0,0], l1.iloc[len(l1)-1, 0]), (l2.iloc[0,0], l2.iloc[len(l2)-1, 0])]

    merged_cf = l1.merge(l2, how = 'outer', on = date_column_name)
    merged_cf = merged_cf.sort_values(date_column_name).reset_index().iloc[:,1:]
    
    # clean up NaN using convert_NaN, first for loan 1 and then for loan 2
    for count in range(2):
        first_day, last_day = F_L_days[count][0], F_L_days[count][1]
        for idx in range(1,len(column_keys)):
            cf_item = columns[column_keys[idx]]
            to_be_cleaned = merged_cf.iloc[:,[0,idx + count* (column_length-1)]].copy()
            clean = convert_NaN(to_be_cleaned, first_day, last_day, cf_item)
            merged_cf.iloc[:, idx + count* (column_length-1)] = clean
    
    # add numbers together
    
    for idx in range(1,column_length):
        new_column_name = column_keys[idx]
        merged_cf[new_column_name] = merged_cf.iloc[:,idx] + merged_cf.iloc[:,idx + column_length - 1]
    
    cols_to_delete = merged_cf.columns[1:column_length*2-1]
    merged_cf.drop(cols_to_delete, axis = 1, inplace=True)
    
    return merged_cf


# In[39]:

def hurdle_cash(cashflow, hurdle):
    
    """
    cashflow : date and cashflow (net cash after mgnt fees) in dataframe
    
    returns amount of cash needed to achieve hurdle rate
    if there is not enough cash, then returns all the cash left
    
    """
    
    IRR_achieved = calc_xirr(cashflow)
    
    if IRR_achieved < hurdle:
        cf_for_LP = cashflow.iloc[-1,1] # if hurdle cannot be achieved all the cash goes to LP
        return (False,cf_for_LP, 0)
    
    else:
        
        LP_cash0 = -sum(cashflow.iloc[:-1,1])
        LP_cash1 = cashflow.iloc[-1,1]        
        cf1 = [(cashflow.iloc[idx,0], cashflow.iloc[idx,1]) for idx in range(len(cashflow))]
        cf0 = cf1.copy()
        cf0[-1] = (cf0[-1][0], LP_cash0)

        iter = 0
        ipsilon = 0.0001
        inv_size = - cf0[0][1]
                
        while (iter < 30) & (xnpv(hurdle, cf1) > 0):
            
            if abs(xnpv(hurdle,cf1))/float(inv_size) > ipsilon:
                
                y0 = xnpv(hurdle,cf0)
                y1 = xnpv(hurdle,cf1)
                LP_cash_new = LP_cash0 + (LP_cash1 - LP_cash0) * (-y0) / (y1 - y0)
                cf_new = cf1.copy()
                cf_new[-1] = (cf_new[-1][0], LP_cash_new)
                y_new = xnpv(hurdle, cf_new)
                
                if y_new >= 0:
                    cf1 = cf_new.copy()
                    LP_cash1 = cf1[-1][1]
                else:
                    cf0 = cf_new.copy()
                    LP_cash0 = cf0[-1][1]
                iter += 1
            else:
                break
            
            available_cash_for_catchup = cashflow.iloc[-1,1] - cf_new[-1][1]
            
        return (True, cf_new, available_cash_for_catchup)


# In[36]:

def calc_catchup(cash, catchup, target, carried_interest):
    """
    target must be 'cumulative cash' * carried interest + 'carried interest paid' - note carried interests recorded negative
    catchup is most commonly 1.00, but may be 0.7 or 0.3
    
    """
    
    finished = False # indicate if the catch up was finished or not
    
    preliminary_to_GP = cash * catchup
    
    if preliminary_to_GP < target:
        # the catch up cannot be finished, so GP receives catch up amount, LP receives the rest
        to_GP = preliminary_to_GP
        to_LP = cash * (1 - catchup)
    
    else: # there is enough cash to finish the catch up
        finished = True
        to_GP = target
        to_LP = cash - target
    
    return (finished, to_GP, to_LP)


# In[33]:

def fee_extraction(cf,
                  mgnt_fee,
                  carried_interest,
                  hurdle = 0.05,
                  catch_up = 1.0,
                  freq = '3M',
                  carry_basis = 'net'):
    
    """
    cf : column 0 : 'date'
         column 1 : 'notional'
         column 2 : 'invested'
         column 3 : 'income'
         column 4 : 'cash flow of principals'
         column 5 : 'net CF'
    freq : fee calculation frequency defaulted to '3M' or quarterly
    """
    col_names = list(cf.columns)
    date_col = col_names[0]
    days = cf.iloc[:, 0].diff().dt.days
    days.values[0] = 0 # change the first row of days from NaN to 0 - days[0] = 0 will produce a warning
    invested = cf.iloc[:, 2]
    days_fraction = days/365
    principals = cf.iloc[:,4]
    cf_before_fees = cf.iloc[:,5]
    cf['mgnt paid'] = - invested * mgnt_fee * days_fraction
    cf['cf after mgnt fees'] =  cf_before_fees + cf['mgnt paid'] 
    cf['cumulative cash'] = cf['cf after mgnt fees'].cumsum() # when this is positive catch up may start
    cf.loc[0,'cumulative cash'] = 0 # change the first row of the cumulative cash from NaN to 0
    
    # carried interest will be % of non-principal amount cash flow net of management fees
    # creat cash flow without principal amount, column 1 has principal before the beginning and column 2
    # has principal at the end of the period - increase of principal was counted as negative in 'cf after magnt fees'
    # we aggregate all the 'gains', which would be the basis for carried interest
    
    # add columns to account for carried interests
    cf.loc[:,'carried interest paid'] = 0
    cf.loc[:,'cf after carried interest'] = cf.loc[:,'cf after mgnt fees']
    
    # find the row where cumulative cash turns positive
    # there can be multiple rows where cash flow would be positive but we just take the first one
    idx = [index for index in range(len(all_loans['cumulative cash'])) if (all_loans['cumulative cash'] > 0)[index]][0]
    
    still_hurdle = True
    still_catchup = True
    
    while (idx < len(cf)):
             
        current_cf = cf.loc[idx:, 'cf after mgnt fees'].values[0]
        
        if still_hurdle:
            (may_achieve, cash_for_LP, cash_for_catchup) = hurdle_cash(cf.loc[:idx,[date_col, 'cf after mgnt fees']],
                                                           hurdle)
            
            catchup_target = cf.loc[idx,'cumulative cash'] * carried_interest +                          cf.loc[:idx, 'carried interest paid'].sum()
                        
            if may_achieve:
                
                (finished, to_GP, to_LP) = calc_catchup(cash_for_catchup, 
                                                            catch_up, 
                                                            catchup_target, 
                                                            carried_interest)
                cf.loc[idx:, 'carried interest paid'] = - to_GP
                cf.loc[idx:, 'cf after carried interest'] = current_cf + cf.loc[idx:, 'carried interest paid']
                still_hurdle = False # they have achieved the hurdles
                    
                    # if catchup is finished, then we do not need to do complex condition analysis
                if finished:
                    still_catchup = False
                        
            # if there was not enough cash to cover all the cash goes to LP so. carried interest paid and 'cf after
            # carried interest remain the same
            else: # if there is not enough cash to achieve the hurdle, all cash goes to LP
                cf.loc[idx:,'cf after carried interest'] = cf.loc[idx:,'cf after mgnt fees']
            
            # if hurdle is not to be achieved, carried interest remains 0 and 'cf after carried interest' remains the same
        elif still_catchup:
            catchup_target = cf.loc[idx,'cumulative cash'] * carried_interest +                          cf.loc[:idx, 'carried interest paid'].sum()
            (finished, to_GP, to_LP) = calc_catchup(cash_for_catchup, 
                                                    catchup, 
                                                    cathup_target, 
                                                    carried_interest)
            cf.loc[idx:, 'carried interest paid'] = - to_GP
            cf.loc[idx:, 'cf after carried interest'] = current_cf + cf.loc[idx:, 'carried interest paid']
            
            if finished:
                still_catchup = False
                
        else: # in case catch up is finished the gains will be shared according to the carried interest rule
                
            cf.loc[idx, 'carried interest paid'] = - current_cf * carried_interest
            cf.loc[idx, 'cf after carried interest'] = current_cf + cf.loc[idx, 'carried interest paid']
            
            # there may /may not be enough cash to finish catchup

            # move one step down
        idx += 1
    
    return cf

