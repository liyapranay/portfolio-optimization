import backtrader as bt

class Zerodha(bt.CommInfoBase):
    """Zerodha brokerage logic for different asset classes and product types."""
    def __init__(self, product_type='CNC', asset_class='equity'):
        super().__init__()
        self.product_type = product_type.upper()
        self.asset_class = asset_class.lower()

    def getcommission(self, size, price, pseudoexec=False):
        value = abs(size) * price
        if pseudoexec:
            return 0.0

        if self.asset_class == 'equity':
            if self.product_type == 'CNC':
                # Delivery (CNC) commissions for equity
                brokerage = min(0.0003 * value, 20)
                stt = 0.00025 * value if size < 0 else 0
                trans_charges = 0.00003 * value
                sebi = 0.000001 * value
                gst = 0.18 * (brokerage + sebi + trans_charges)
                stamp_duty = 0.00015 * value if size > 0 else 0  # Stamp duty on buy
                return brokerage + stt + trans_charges + sebi + gst + stamp_duty
            elif self.product_type == 'MIS':
                # Intraday (MIS) commissions for equity
                brokerage = min(0.0003 * value, 20)
                stt = 0.00025 * value if size < 0 else 0
                trans_charges = 0.00003 * value
                sebi = 0.000001 * value
                gst = 0.18 * (brokerage + sebi + trans_charges)
                stamp_duty = 0.00003 * value if size > 0 else 0  # Higher stamp duty for intraday
                return brokerage + stt + trans_charges + sebi + gst + stamp_duty
            else:
                raise ValueError(f"Unknown product_type for Zerodha equity: {self.product_type}")
        elif self.asset_class == 'futures':
            # Futures commissions
            brokerage = min(0.0003 * value, 20)
            stt = 0.000125 * value if size < 0 else 0
            trans_charges = 0.000019 * value
            sebi = 0.000001 * value
            gst = 0.18 * (brokerage + sebi + trans_charges)
            stamp_duty = 0.00002 * value if size > 0 else 0
            return brokerage + stt + trans_charges + sebi + gst + stamp_duty
        elif self.asset_class == 'options':
            # Options commissions
            brokerage = min(0.0003 * value, 20)
            stt = 0.0005 * value if size < 0 else 0.000125 * value
            trans_charges = 0.00005 * value
            sebi = 0.000001 * value
            gst = 0.18 * (brokerage + sebi + trans_charges)
            stamp_duty = 0.00003 * value if size > 0 else 0
            return brokerage + stt + trans_charges + sebi + gst + stamp_duty
        else:
            raise ValueError(f"Unknown asset_class for Zerodha: {self.asset_class}")


class Upstox(bt.CommInfoBase):
    """Upstox brokerage logic for different asset classes."""
    def __init__(self, product_type='CNC', asset_class='equity'):
        super().__init__()
        self.product_type = product_type.upper()
        self.asset_class = asset_class.lower()

    def getcommission(self, size, price, pseudoexec=False):
        value = abs(size) * price
        if pseudoexec:
            return 0.0

        if self.asset_class == 'equity':
            brokerage = min(0.0005 * value, 20)
            stt = 0.00025 * value if size < 0 else 0
            trans_charges = 0.00003 * value
            sebi = 0.000001 * value
            gst = 0.18 * (brokerage + sebi + trans_charges)
            stamp_duty = 0.00015 * value if size > 0 else 0
            return brokerage + stt + trans_charges + sebi + gst + stamp_duty
        elif self.asset_class == 'futures':
            brokerage = min(0.0005 * value, 20)
            stt = 0.000125 * value if size < 0 else 0
            trans_charges = 0.000019 * value
            sebi = 0.000001 * value
            gst = 0.18 * (brokerage + sebi + trans_charges)
            stamp_duty = 0.00002 * value if size > 0 else 0
            return brokerage + stt + trans_charges + sebi + gst + stamp_duty
        elif self.asset_class == 'options':
            brokerage = min(0.0005 * value, 20)
            stt = 0.0005 * value if size < 0 else 0.000125 * value
            trans_charges = 0.00005 * value
            sebi = 0.000001 * value
            gst = 0.18 * (brokerage + sebi + trans_charges)
            stamp_duty = 0.00003 * value if size > 0 else 0
            return brokerage + stt + trans_charges + sebi + gst + stamp_duty
        else:
            raise ValueError(f"Unknown asset_class for Upstox: {self.asset_class}")


def get_commission_class(broker_name, product_type='CNC', asset_class='equity'):
    broker_name = broker_name.lower()
    if broker_name == 'zerodha':
        return Zerodha(product_type, asset_class)
    elif broker_name == 'upstox':
        return Upstox(product_type, asset_class)
    else:
        raise ValueError(f"Unknown broker name: {broker_name}")