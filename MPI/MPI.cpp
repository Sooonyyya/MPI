#define USE_MPI

#ifdef USE_MPI
#include <mpi.h>
#else
#include <omp.h>
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>
#include <chrono>
#include <algorithm>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

unordered_map<string, int> getFrequenciesFromFile(const string& filename)
{
    ifstream file(filename, ios::in | ios::binary);
    if (!file.is_open()) return {};

    unordered_map<string, int> frequencies;
    istringstream stream((ostringstream() << file.rdbuf()).str());
    string token;

    while (stream >> token)
    {
        string word;
        for (char c : token)
            if (isalpha((unsigned char)c))
                word += tolower((unsigned char)c);
        if (!word.empty())
            frequencies[word]++;
    }

    return frequencies;
}

void mergeMaps(unordered_map<string, int>& dst, const unordered_map<string, int>& src)
{
    for (const auto& [word, count] : src)
        dst[word] += count;
}

void findFiles(const fs::path& basePath, vector<string>& found)
{
    try {
        if (fs::is_regular_file(basePath))
            found.push_back(basePath.string());
        else
            for (const auto& entry : fs::recursive_directory_iterator(
                basePath, fs::directory_options::skip_permission_denied))
                if (entry.is_regular_file())
                    found.push_back(entry.path().string());
    }
    catch (const fs::filesystem_error& e) {
        cerr << "[ERROR] " << e.what() << endl;
    }
}

void printTopWords(const unordered_map<string, int>& freq, int limit = 10)
{
    vector<pair<string, int>> sorted(freq.begin(), freq.end());
    sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
        });
    for (int i = 0; i < min(limit, (int)sorted.size()); i++)
        cout << sorted[i].first << " : " << sorted[i].second << endl;
    cout << "Unique words: " << freq.size() << endl;
}

unordered_map<string, int> singleThreadWordCount(const vector<string>& files)
{
    unordered_map<string, int> global;
    for (const auto& f : files)
        mergeMaps(global, getFrequenciesFromFile(f));
    return global;
}

#ifndef USE_MPI

unordered_map<string, int> openMpWordCount(const vector<string>& files, int threadsCount)
{
    omp_set_num_threads(threadsCount);
    int actualThreads = 0;
    vector<unordered_map<string, int>> localMaps;

#pragma omp parallel
    {
#pragma omp single
        {
            actualThreads = omp_get_num_threads();
            localMaps.resize(actualThreads);
        }

        int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic)
        for (int i = 0; i < (int)files.size(); i++)
            mergeMaps(localMaps[tid], getFrequenciesFromFile(files[i]));
    }

    unordered_map<string, int> global;
    for (const auto& local : localMaps)
        mergeMaps(global, local);
    return global;
}

#else

string serializeMap(const unordered_map<string, int>& freq)
{
    ostringstream ss;
    for (const auto& [word, count] : freq)
        ss << word << " " << count << "\n";
    return ss.str();
}

unordered_map<string, int> deserializeMap(const string& data)
{
    unordered_map<string, int> result;
    istringstream ss(data);
    string word; int count;
    while (ss >> word >> count)
        result[word] += count;
    return result;
}

string serializePaths(const vector<string>& paths)
{
    ostringstream ss;
    for (const auto& p : paths) ss << p << "\n";
    return ss.str();
}

vector<string> deserializePaths(const string& data)
{
    vector<string> paths;
    istringstream ss(data);
    string line;
    while (getline(ss, line))
        if (!line.empty()) paths.push_back(line);
    return paths;
}

#endif

int main(int argc, char* argv[])
{
#ifndef USE_MPI

    if (argc < 3) { cerr << "Usage: program.exe <threads> path1 ..." << endl; return 1; }

    int threadsCount = stoi(argv[1]);
    vector<string> files;
    for (int i = 2; i < argc; i++) findFiles(fs::path(argv[i]), files);
    if (files.empty()) { cout << "No files found." << endl; return 0; }

    sort(files.begin(), files.end());
    cout << "Processing " << files.size() << " files..." << endl;

    auto t0 = chrono::high_resolution_clock::now();
    auto singleResult = singleThreadWordCount(files);
    double singleMs = chrono::duration_cast<chrono::microseconds>(
        chrono::high_resolution_clock::now() - t0).count() / 1000.0;

    cout << "=== Single-thread ===" << endl;
    cout << "Execution time: " << singleMs << " ms" << endl;
    printTopWords(singleResult);

    double ompStart = omp_get_wtime();
    auto ompResult = openMpWordCount(files, threadsCount);
    double ompMs = (omp_get_wtime() - ompStart) * 1000.0;

    cout << "\n=== OpenMP ===" << endl;
    cout << "Threads: " << threadsCount << endl;
    cout << "Execution time: " << ompMs << " ms" << endl;
    printTopWords(ompResult);

#else

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<string> files;
    int flag = 0;

    if (rank == 0)
    {
        if (argc < 2) { MPI_Abort(MPI_COMM_WORLD, 1); return 1; }
        for (int i = 1; i < argc; i++) findFiles(fs::path(argv[i]), files);
        sort(files.begin(), files.end());
        flag = !files.empty() ? 1 : 0;
    }

    MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!flag) { MPI_Finalize(); return 0; }

    if (rank == 0)
    {
        auto t0 = chrono::high_resolution_clock::now();
        auto r = singleThreadWordCount(files);
        double ms = chrono::duration_cast<chrono::microseconds>(
            chrono::high_resolution_clock::now() - t0).count() / 1000.0;
        cout << "=== Single-thread ===" << endl;
        cout << "Execution time: " << ms << " ms" << endl;
        printTopWords(r);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double mpiStart = MPI_Wtime();

    vector<int> sendCounts(size, 0), displacements(size, 0);
    string allPaths;

    if (rank == 0)
    {
        vector<vector<string>> parts(size);
        for (int i = 0; i < (int)files.size(); i++)
            parts[i % size].push_back(files[i]);

        int offset = 0;
        for (int i = 0; i < size; i++)
        {
            string s = serializePaths(parts[i]);
            sendCounts[i] = s.size();
            displacements[i] = offset;
            offset += s.size();
            allPaths += s;
        }
    }

    MPI_Bcast(sendCounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displacements.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

    int localLen = sendCounts[rank];
    vector<char> localBuf(max(localLen, 1));
    vector<char> sendBuf(allPaths.begin(), allPaths.end());
    if (sendBuf.empty()) sendBuf.push_back(0);

    MPI_Scatterv(sendBuf.data(), sendCounts.data(), displacements.data(), MPI_CHAR,
        localBuf.data(), localLen, MPI_CHAR, 0, MPI_COMM_WORLD);

    auto localFiles = deserializePaths(string(localBuf.begin(), localBuf.begin() + localLen));
    unordered_map<string, int> localFreq;
    for (const auto& f : localFiles)
        mergeMaps(localFreq, getFrequenciesFromFile(f));

    string serialized = serializeMap(localFreq);
    int localSize = serialized.size();
    vector<int> recvSizes(size, 0);
    MPI_Gather(&localSize, 1, MPI_INT, recvSizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> recvDisp(size, 0);
    int totalSize = 0;
    if (rank == 0)
        for (int i = 0; i < size; i++) { recvDisp[i] = totalSize; totalSize += recvSizes[i]; }

    vector<char> sendMapBuf(serialized.begin(), serialized.end());
    if (sendMapBuf.empty()) sendMapBuf.push_back(0);
    vector<char> recvBuf(max(totalSize, 1));

    MPI_Gatherv(sendMapBuf.data(), localSize, MPI_CHAR,
        recvBuf.data(), recvSizes.data(), recvDisp.data(), MPI_CHAR,
        0, MPI_COMM_WORLD);

    double mpiMs = (MPI_Wtime() - mpiStart) * 1000.0;

    if (rank == 0)
    {
        unordered_map<string, int> global;
        for (int i = 0; i < size; i++)
            if (recvSizes[i] > 0)
                mergeMaps(global, deserializeMap(string(recvBuf.data() + recvDisp[i], recvSizes[i])));

        cout << "\n=== MPI ===" << endl;
        cout << "Processes: " << size << endl;
        cout << "Execution time: " << mpiMs << " ms" << endl;
        printTopWords(global);
    }

    MPI_Finalize();
#endif

    return 0;
}