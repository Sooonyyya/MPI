//#define USE_MPI

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

using namespace std;

string cleanWord(const string& word)
{
    string result;
    result.reserve(word.size());

    for (char c : word)
    {
        if (isalpha((unsigned char)c))
            result += (char)tolower((unsigned char)c);
    }

    return result;
}

unordered_map<string, int> countWordsInFile(const string& filename)
{
    unordered_map<string, int> frequencies;
    ifstream file(filename);

    if (!file.is_open())
    {
#pragma omp critical
        cerr << "[ERROR] Cannot open file: " << filename << endl;
        return frequencies;
    }

    string word;
    while (file >> word)
    {
        string cleaned = cleanWord(word);
        if (!cleaned.empty())
            frequencies[cleaned]++;
    }

    return frequencies;
}

void mergeMaps(unordered_map<string, int>& globalMap,
    const unordered_map<string, int>& localMap)
{
    for (const auto& pair : localMap)
        globalMap[pair.first] += pair.second;
}

void printTopWords(const unordered_map<string, int>& frequencies, int limit = 10)
{
    vector<pair<string, int>> sorted(frequencies.begin(), frequencies.end());

    sort(sorted.begin(), sorted.end(),
        [](const auto& a, const auto& b)
        {
            return a.second > b.second;
        });

    int printed = 0;

    for (const auto& pair : sorted)
    {
        cout << pair.first << " : " << pair.second << endl;

        printed++;
        if (printed >= limit)
            break;
    }

    cout << "Unique words: " << frequencies.size() << endl;
}

unordered_map<string, int> singleThreadWordCount(const vector<string>& files)
{
    unordered_map<string, int> globalFrequencies;

    for (const string& file : files)
    {
        auto fileFrequencies = countWordsInFile(file);
        mergeMaps(globalFrequencies, fileFrequencies);
    }

    return globalFrequencies;
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
            if (actualThreads > 0)
            {
                localMaps.resize(actualThreads);
            }
            else
            {
#pragma omp critical
                cerr << "[ERROR] OpenMP: 0 actual threads." << endl;
            }
        }

        if (actualThreads > 0)
        {
            int threadId = omp_get_thread_num();

            if (threadId >= 0 && threadId < actualThreads)
            {
#pragma omp for schedule(dynamic)
                for (int i = 0; i < (int)files.size(); i++)
                {
                    auto fileFrequencies = countWordsInFile(files[i]);
                    mergeMaps(localMaps[threadId], fileFrequencies);
                }
            }
            else
            {
#pragma omp critical
                cerr << "[ERROR] OpenMP: invalid thread ID " << threadId << endl;
            }
        }
    }

    unordered_map<string, int> globalFrequencies;

    for (const auto& localMap : localMaps)
        mergeMaps(globalFrequencies, localMap);

    return globalFrequencies;
}

#else

string serializeMap(const unordered_map<string, int>& frequencies)
{
    stringstream ss;

    for (const auto& pair : frequencies)
        ss << pair.first << "\t" << pair.second << "\n";

    return ss.str();
}

unordered_map<string, int> deserializeMap(const string& data)
{
    unordered_map<string, int> frequencies;
    istringstream ss(data);
    string line;

    while (getline(ss, line))
    {
        if (line.empty())
            continue;

        size_t tab = line.find('\t');

        if (tab == string::npos)
            continue;

        string word = line.substr(0, tab);

        try
        {
            int count = stoi(line.substr(tab + 1));
            if (!word.empty())
                frequencies[word] += count;
        }
        catch (...) {}
    }

    return frequencies;
}

string serializePaths(const vector<string>& paths)
{
    stringstream ss;

    for (const auto& path : paths)
        ss << path << "\n";

    return ss.str();
}

vector<string> deserializePaths(const string& data)
{
    vector<string> paths;
    istringstream ss(data);
    string line;

    while (getline(ss, line))
    {
        if (!line.empty())
            paths.push_back(line);
    }

    return paths;
}

#endif

int main(int argc, char* argv[])
{
#ifndef USE_MPI

    if (argc < 3)
    {
        cerr << "Usage: program.exe <threads> file1.txt file2.txt ..." << endl;
        return 1;
    }

    int threadsCount = stoi(argv[1]);

    vector<string> files;
    for (int i = 2; i < argc; i++)
        files.emplace_back(argv[i]);

    auto singleStart = chrono::high_resolution_clock::now();
    auto singleResult = singleThreadWordCount(files);
    auto singleEnd = chrono::high_resolution_clock::now();

    double singleTime =
        chrono::duration_cast<chrono::microseconds>(singleEnd - singleStart).count() / 1000.0;

    cout << "=== Single-thread ===" << endl;
    cout << "Execution time: " << singleTime << " ms" << endl;
    printTopWords(singleResult);
    cout << endl;

    double ompStart = omp_get_wtime();
    auto ompResult = openMpWordCount(files, threadsCount);
    double ompEnd = omp_get_wtime();

    cout << "=== OpenMP ===" << endl;
    cout << "Threads: " << threadsCount << endl;
    cout << "Execution time: " << (ompEnd - ompStart) * 1000.0 << " ms" << endl;
    printTopWords(ompResult);
    cout << endl;

#else

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<string> files;

    if (rank == 0)
    {
        if (argc < 2)
        {
            cerr << "Usage: mpiexec -n <processes> program.exe file1.txt file2.txt ..." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        for (int i = 1; i < argc; i++)
            files.emplace_back(argv[i]);

        auto singleStart = chrono::high_resolution_clock::now();
        auto singleResult = singleThreadWordCount(files);
        auto singleEnd = chrono::high_resolution_clock::now();

        double singleTime =
            chrono::duration_cast<chrono::microseconds>(singleEnd - singleStart).count() / 1000.0;

        cout << "=== Single-thread ===" << endl;
        cout << "Execution time: " << singleTime << " ms" << endl;
        printTopWords(singleResult);
        cout << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double mpiStart = MPI_Wtime();

    vector<int> sendCounts(size, 0);
    vector<int> displacements(size, 0);
    string allSerializedPaths;

    if (rank == 0)
    {
        vector<vector<string>> partitions(size);

        for (int i = 0; i < (int)files.size(); i++)
            partitions[i % size].push_back(files[i]);

        int offset = 0;

        for (int i = 0; i < size; i++)
        {
            string part = serializePaths(partitions[i]);
            sendCounts[i] = (int)part.size();
            displacements[i] = offset;
            offset += (int)part.size();
            allSerializedPaths += part;
        }
    }

    MPI_Bcast(sendCounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displacements.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

    int localLen = sendCounts[rank];
    vector<char> localBuf(localLen > 0 ? localLen : 1);

    vector<char> sendBuf(allSerializedPaths.begin(), allSerializedPaths.end());
    if (sendBuf.empty())
        sendBuf.push_back(0);

    MPI_Scatterv(
        sendBuf.data(),
        sendCounts.data(),
        displacements.data(),
        MPI_CHAR,
        localBuf.data(),
        localLen,
        MPI_CHAR,
        0,
        MPI_COMM_WORLD
    );

    string localPathsStr(localBuf.begin(), localBuf.begin() + localLen);
    vector<string> localFiles = deserializePaths(localPathsStr);

    unordered_map<string, int> localFrequencies;

    for (const string& filename : localFiles)
    {
        auto fileFrequencies = countWordsInFile(filename);
        mergeMaps(localFrequencies, fileFrequencies);
    }

    string serializedLocalMap = serializeMap(localFrequencies);
    int localSize = (int)serializedLocalMap.size();

    vector<int> receiveSizes(size, 0);

    MPI_Gather(
        &localSize, 1, MPI_INT,
        receiveSizes.data(), 1, MPI_INT,
        0, MPI_COMM_WORLD
    );

    vector<int> recvDisplacements(size, 0);
    int totalSize = 0;

    if (rank == 0)
    {
        for (int i = 0; i < size; i++)
        {
            recvDisplacements[i] = totalSize;
            totalSize += receiveSizes[i];
        }
    }

    vector<char> mapSendBuf(serializedLocalMap.begin(), serializedLocalMap.end());
    if (mapSendBuf.empty())
        mapSendBuf.push_back(0);

    vector<char> receiveBuffer(totalSize > 0 ? totalSize : 1);

    MPI_Gatherv(
        mapSendBuf.data(), localSize, MPI_CHAR,
        receiveBuffer.data(), receiveSizes.data(), recvDisplacements.data(), MPI_CHAR,
        0, MPI_COMM_WORLD
    );

    double mpiEnd = MPI_Wtime();

    if (rank == 0)
    {
        unordered_map<string, int> globalMpiFrequencies;

        for (int i = 0; i < size; i++)
        {
            if (receiveSizes[i] == 0)
                continue;

            string part(
                receiveBuffer.data() + recvDisplacements[i],
                receiveSizes[i]
            );

            auto tempMap = deserializeMap(part);
            mergeMaps(globalMpiFrequencies, tempMap);
        }

        cout << "=== MPI ===" << endl;
        cout << "Processes: " << size << endl;
        cout << "Execution time: " << (mpiEnd - mpiStart) * 1000.0 << " ms" << endl;
        printTopWords(globalMpiFrequencies);
        cout << endl;
    }

    MPI_Finalize();

#endif

    return 0;
}