#include <iostream>
#include <numeric>
#include <unistd.h>

#include "../../include/utils/time_utils.h"
// #include "utils/time_utils.h"

timelap_counter::timelap_counter(int num_counter)
{
    for (int i = 0; i < num_counter; i++) {
        timespec t;
        laps.push_back(t);
    }
    laps_length = num_counter;
}

void timelap_counter::timelap()
{
    clock_gettime(CLOCK_REALTIME, &laps[current_lap_id]);
    current_lap_id += 1;
}

void timelap_counter::timelap(int lap_id)
{
    clock_gettime(CLOCK_REALTIME, &laps[lap_id]);
    current_lap_id = lap_id;
}

void timelap_counter::add_lap()
{
    timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    laps.push_back(t);
    laps_length += 1;
}

double timelap_counter::milisec_between(int start_id, int end_id)
{
    long sec;
    long nsec;
    sec  = laps[end_id].tv_sec - laps[start_id].tv_sec;
    nsec = laps[end_id].tv_nsec - laps[start_id].tv_nsec;
    return (double)sec * 1000 + (double)nsec / (1000 * 1000);
}

void stopwatch::start()
{
    is_on = true;
    clock_gettime(CLOCK_REALTIME, &stime);
}

void stopwatch::lap()
{
    if (is_on) {
        clock_gettime(CLOCK_REALTIME, &etime);
        laps.push_back(calc_timems());
        ++laps_len;
        stime = etime;
    }
}

void stopwatch::stop()
{
    if (is_on) {
        clock_gettime(CLOCK_REALTIME, &etime);
        laps.push_back(calc_timems());
        ++laps_len;
        is_on = false;
    }
}

void stopwatch::reset()
{
    is_on = false;
    laps.clear();
    laps_len = 0;
}

double stopwatch::get_lap(int lap_id) { return laps[lap_id]; }

double stopwatch::get_total()
{
    return std::accumulate(laps.begin(), laps.end(), 0.0);
}

double stopwatch::calc_timems()
{
    long sec;
    long nsec;
    sec  = etime.tv_sec - stime.tv_sec;
    nsec = etime.tv_nsec - stime.tv_nsec;
    return (double)sec * 1000 + (double)nsec / (1000 * 1000);
}

// int main() {
//     double elapsed_time_1, elapsed_time_2, elapsed_time_3;
//     timelap_counter stop_watch{3};
//     stop_watch.timelap();
//     sleep(1);
//     stop_watch.timelap();
//     sleep(1);
//     stop_watch.timelap();
//     elapsed_time_1 = stop_watch.milisec_between(0, 1);
//     elapsed_time_2 = stop_watch.milisec_between(1, 2);
//     elapsed_time_3 = stop_watch.milisec_between(0, 2);
//     std::cout << elapsed_time_1 << std::endl;
//     std::cout << elapsed_time_2 << std::endl;
//     std::cout << elapsed_time_3 << std::endl;

//     stopwatch sw;
//     sw.start();
//     sleep(1);
//     sw.stop();
//     sleep(2);
//     sw.start();
//     sleep(4);
//     sw.lap();
//     sleep(8);
//     sw.stop();
//     for (auto t : sw.laps) {
//         std::cout << t << std::endl;
//     }
//     std::cout << sw.get_lap(1) << std::endl;
//     std::cout << sw.get_total() << std::endl;
// }