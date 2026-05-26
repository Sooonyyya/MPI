#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>
#include <chrono>

using namespace std;

string cleanWord(string word)
{
    string result;

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
        cerr << "Cannot open file: " << filename << endl;
        return frequencies;
    }

    string word;

    while (file >> word)
    {
        word = cleanWord(word);

        if (!word.empty())
            frequencies[word]++;
    }

    return frequencies;
}

void mergeMaps(unordered_map<string, int>& globalMap, const unordered_map<string, int>& localMap)
{
    for (const auto& pair : localMap)
        globalMap[pair.first] += pair.second;
}

unordered_map<string, int> singleThreadWordCount(const vector<string>& files)
{
    unordered_map<string, int> globalFrequencies;

    for (const string& file : files)
    {
        unordered_map<string, int> fileFrequencies = countWordsInFile(file);
        mergeMaps(globalFrequencies, fileFrequencies);
    }

    return globalFrequencies;
}

unordered_map<string, int> openMpWordCount(const vector<string>& files, int threadsCount)
{
    omp_set_num_threads(threadsCount);

    vector<unordered_map<string, int>> localMaps(threadsCount);

#pragma omp parallel
    {
        int threadId = omp_get_thread_num();

#pragma omp for schedule(dynamic)
        for (int i = 0; i < (int)files.size(); i++)
        {
            unordered_map<string, int> fileFrequencies = countWordsInFile(files[i]);
            mergeMaps(localMaps[threadId], fileFrequencies);
        }
    }

    unordered_map<string, int> globalFrequencies;

    for (const auto& localMap : localMaps)
        mergeMaps(globalFrequencies, localMap);

    return globalFrequencies;
}

string serializeMap(const unordered_map<string, int>& frequencies)
{
    stringstream ss;

    for (const auto& pair : frequencies)
        ss << pair.first << " " << pair.second << "\n";

    return ss.str();
}

unordered_map<string, int> deserializeMap(const string& data)
{
    unordered_map<string, int> frequencies;
    stringstream ss(data);

    string word;
    int count;

    while (ss >> word >> count)
        frequencies[word] += count;

    return frequencies;
}

void printFirstWords(const unordered_map<string, int>& frequencies, int limit = 30)
{
    int printed = 0;

    for (const auto& pair : frequencies)
    {
        cout << pair.first << " : " << pair.second << endl;

        printed++;

        if (printed >= limit)
            break;
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<string> files =
    {
        "texts/file1.txt",
        "texts/file2.txt",
        "texts/file3.txt"
    };

    if (rank == 0)
    {
        auto singleStart = chrono::high_resolution_clock::now();
        unordered_map<string, int> singleResult = singleThreadWordCount(files);
        auto singleEnd = chrono::high_resolution_clock::now();

        double singleTime = chrono::duration_cast<chrono::microseconds>(
            singleEnd - singleStart
        ).count() / 1000.0;

        cout << "Single-thread word frequency result:" << endl;
        cout << "Execution time: " << singleTime << " ms" << endl;
        cout << endl;

        auto ompStart = chrono::high_resolution_clock::now();
        unordered_map<string, int> ompResult = openMpWordCount(files, size);
        auto ompEnd = chrono::high_resolution_clock::now();

        double ompTime = chrono::duration_cast<chrono::microseconds>(
            ompEnd - ompStart
        ).count() / 1000.0;

        cout << "OpenMP word frequency result:" << endl;
        cout << "Threads: " << size << endl;
        cout << "Execution time: " << ompTime << " ms" << endl;
        cout << endl;

        cout << "First words:" << endl;
        printFirstWords(ompResult);
        cout << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double mpiStart = MPI_Wtime();

    unordered_map<string, int> localFrequencies;

    for (int i = rank; i < (int)files.size(); i += size)
    {
        unordered_map<string, int> fileFrequencies = countWordsInFile(files[i]);
        mergeMaps(localFrequencies, fileFrequencies);
    }

    string serializedLocalMap = serializeMap(localFrequencies);
    int localSize = (int)serializedLocalMap.size();

    vector<int> receiveSizes(size);

    MPI_Gather(
        &localSize,
        1,
        MPI_INT,
        receiveSizes.data(),
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    vector<int> displacements(size);
    int totalSize = 0;

    if (rank == 0)
    {
        for (int i = 0; i < size; i++)
        {
            displacements[i] = totalSize;
            totalSize += receiveSizes[i];
        }
    }

    vector<char> receiveBuffer(totalSize);

    MPI_Gatherv(
        serializedLocalMap.data(),
        localSize,
        MPI_CHAR,
        receiveBuffer.data(),
        receiveSizes.data(),
        displacements.data(),
        MPI_CHAR,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);

    double mpiEnd = MPI_Wtime();

    if (rank == 0)
    {
        unordered_map<string, int> globalMpiFrequencies;

        for (int i = 0; i < size; i++)
        {
            string part(
                receiveBuffer.data() + displacements[i],
                receiveSizes[i]
            );

            unordered_map<string, int> tempMap = deserializeMap(part);
            mergeMaps(globalMpiFrequencies, tempMap);
        }

        cout << "MPI word frequency result:" << endl;
        cout << "Processes: " << size << endl;
        cout << "Execution time: " << (mpiEnd - mpiStart) * 1000 << " ms" << endl;
        cout << endl;

        cout << "First words:" << endl;
        printFirstWords(globalMpiFrequencies);
        cout << endl;
    }

    MPI_Finalize();

    return 0;
}