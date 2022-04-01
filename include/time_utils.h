#ifndef TIME_UTILS_H
#define TIME_UTILS_H

#include <time.h>
#include <vector>

struct timelap_counter {
    std::vector<timespec> laps;
    int laps_length = 0;
    int current_lap_id = 0;

    timelap_counter(int num_counter);
    void timelap();
    void timelap(int lap_id);
    void add_lap();
    double milisec_between(int start_id, int end_id);
};

#endif