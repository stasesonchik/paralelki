#include <iostream>
#include <queue>
#include <future>
#include <thread>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <condition_variable>
#include <random>
#include <fstream>
#include <atomic>

template<typename T>
class Server {
private:
    std::queue<std::pair<size_t, std::function<T()>>> task_queue;
    std::unordered_map<size_t, T> task_results;

    std::mutex task_mutex;
    std::mutex result_mutex;
    std::condition_variable task_cond;
    std::condition_variable result_cond;

    std::thread worker_thread;
    std::atomic<bool> running;
    size_t task_counter;
    
    void process_tasks()  {
        while (running) {
            std::unique_lock<std::mutex> lock(task_mutex);
            task_cond.wait(lock, [&]() { return !task_queue.empty() || !running; });

            if (!running && task_queue.empty()) return;

            auto [id, task] = task_queue.front();
            task_queue.pop();
            lock.unlock();

            T result;
            try {
                result = task();
            } catch (...) {
                continue;
            }

            std::unique_lock<std::mutex> result_lock(result_mutex);
            task_results[id] = result;
            result_cond.notify_all();
        }
    }

public:
    Server() :  running(false), task_counter(0) {}
    ~Server() {
        stop();
    }

    void start() {
        running = true;
        worker_thread = std::thread(&Server::process_tasks, this);
    }
    
    void stop() {
        if (running) {
            running = false;
            task_cond.notify_all();
            worker_thread.join();
        }
    }
    
    size_t add_task(std::function<T()> task) {
        std::unique_lock<std::mutex> lock(task_mutex);
        size_t id = task_counter++;
        task_queue.push({id, task});
        task_cond.notify_one();
        return id;
    }
    
    T request_result(size_t id) {
        std::unique_lock<std::mutex> lock(result_mutex);
        result_cond.wait(lock, [&]() { return task_results.find(id) != task_results.end(); });
        T result = task_results[id];
        task_results.erase(id); 
        return result;
    }
};

void client_sin(Server<double>& server, int task_count, const std::string& filename) {
    std::ofstream file(filename);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, M_PI);
    std::unordered_map<size_t, double> task_data; 

    std::vector<size_t> task_ids;
    for (int i = 0; i < task_count; ++i) {
        double arg = dist(gen);
        size_t id = server.add_task([arg]() { return std::sin(arg); });
        task_ids.push_back(id);
        task_data[id] = arg;
    }

    for (size_t id : task_ids) {
        double arg = task_data[id];
        double result = server.request_result(id);
        file << "id: " << id << ", arg: " << arg << ", sin: " << result << "\n";
    }
    file.close();
}

void client_sqrt(Server<double>& server, int task_count, const std::string& filename) {
    std::ofstream file(filename);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(1.0, 100.0);
    std::unordered_map<size_t, double> task_data; 

    std::vector<size_t> task_ids;
    for (int i = 0; i < task_count; ++i) {
        double arg = dist(gen);
        size_t id = server.add_task([arg]() { return std::sqrt(arg); });
        task_ids.push_back(id);
        task_data[id] = arg;
    }

    for (size_t id : task_ids) {
        double arg = task_data[id];
        double result = server.request_result(id);
        file << "id: " << id << ", arg: " << arg << ", sqrt: " << result << "\n";
    }
    file.close();
}

void client_pow(Server<double>& server, int task_count, const std::string& filename) {
    std::ofstream file(filename);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist1(1.0, 10.0);
    std::uniform_real_distribution<> dist2(1.0, 5.0);

    std::unordered_map<size_t, std::pair<double, double>> task_data;

    std::vector<size_t> task_ids;
    for (int i = 0; i < task_count; ++i) {
        double base = dist1(gen);
        double exponent = dist2(gen);
        size_t id = server.add_task([base, exponent]() { return std::pow(base, exponent); });
        task_ids.push_back(id);
        task_data[id] = {base, exponent};
    }

    for (size_t id : task_ids) {
        auto [base, exponent] = task_data[id];
        double result = server.request_result(id);
        file << "id: " << id << " base: " << base << " exponent: " << exponent << " pow: " << result << "\n";
    }

    file.close();
}

int main() {
    Server<double> server;
    server.start();

    int task_count = 100;

    std::thread t1(client_sin, std::ref(server), task_count, "sin_results.txt");
    std::thread t2(client_sqrt, std::ref(server), task_count, "sqrt_results.txt");
    std::thread t3(client_pow, std::ref(server), task_count, "pow_results.txt");

    t1.join();
    t2.join();
    t3.join();

    server.stop();
    return 0;
}
