from dataclasses import dataclass


class Signal:
    def __init__(self):
        self.io_input = []
        self.io_ouput = []
        self.alarm_values = [0, 0, 0, 0, 0, 0]
        self.sample_values = [0, 0, 0, 0, 0, 0]
        self.values = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def check_alarm(self, df, instrument_supress, time_, i, v, set_point_ll=-67 * 100):

        self.alarm_values.append(1.0 if v <= set_point_ll else 0.0)
        self.alarm_values.pop(0)
        self.sample_values.append(v)
        self.sample_values.pop(0)

        check = 0

        if sum(self.alarm_values) == 6:

            # select data for supression
            start = time_ - timedelta(seconds=120)
            end = time_

            # code repetition for n supressions
            mask = (df[instrument_supress].notnull()) & (
                df['time'] >= start) & (df['time'] <= end)
            supression = df.loc[mask][instrument_supress]

            print(supression)
            print_colored(f'{start} , {end}', Color.BLUE)

            # change check to 1 if supression is enable
            if not checkSupress(supression, time_):
                check = 1

            # code repetition for n supressions

            # print alarm
            color = Color.RED if check == 1 else Color.GREEN
            print_colored('iter = ' + str(i), color)
            print_colored('time = ' + str(time_), color)
            print_colored('signal inputs ' + str(self.values) + ' kPa', color)
            print_colored('signal outputs' +
                          str(self.sample_values) + ' kPa/min', color)
            self.alarm_values = [0, 0, 0, 0, 0, 0]
            self.sample_values = [0, 0, 0, 0, 0, 0]

        return check

    def signalFilter(self, df, instrument, instrument_supress, coef=[-4, -3, -2, -1, 0, 1, 2, 3, 4], gain=1.0):

        delta_sec = 5
        mask = df[instrument].notnull()
        current_data = df.loc[mask][instrument]
        current_time = df.loc[mask]['time']

        total_iterations = int(len(current_data) * 1)
        iter_out = []
        status_in_alarm = 0
        alarms = []

        print_colored('total iterations ' + str(total_iterations), Color.BLUE)

        for i in range(total_iterations):
            iter_out.append(i)
            self.io_input.append(current_data.iloc[i])
            out_ = 0
            self.values.append(current_data.iloc[i])
            self.values.pop(0)
            for c, v in zip(coef, self.values):
                out_ += c * v
            filtered = gain * out_ / delta_sec
            self.io_ouput.append(filtered)
            alarms.append(self.check_alarm(df, instrument_supress,
                          current_time.iloc[i], i, filtered))

        # plot fig size
        size = (18, 6)
        # alarm
        alm = DataFrame({'time': iter_out, 'alarms': alarms})
        alm.plot(x='time', y='alarms', kind='line', figsize=size)

        # input signal
        df_input = DataFrame({'time': iter_out, instrument: self.io_input})
        df_input.plot(x='time', y=instrument, kind='line', figsize=size)

        # supress signal
        mask = df[instrument_supress].notnull()
        df.loc[mask].plot(x='time', y=instrument_supress,
                          kind='line', figsize=size)

        # filtered signal
        result = DataFrame(
            {'time': iter_out, 'filtered_signal': self.io_ouput})
        result.plot(x='time', y='filtered_signal', kind='line', figsize=size)

        # print alarms
        total_alarms = alm.sum().values[1]
        color = Color.RED if total_alarms >= 1 else Color.GREEN
        print_colored('total alarms =  ' + str(total_alarms), color)

        return result
