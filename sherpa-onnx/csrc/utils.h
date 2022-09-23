#include <iostream>
#include <fstream>


void vector2file(std::vector<float> vector, std::string saveFileName){
    std::ofstream f(saveFileName);
    for(std::vector<float>::const_iterator i = vector.begin(); i != vector.end(); ++i) {
        f << *i << '\n';
    }
}


std::vector<std::string> hyps2result(std::map<int, std::string> token_map, std::vector<std::vector<int32_t>> hyps, int context_size = 2){
    std::vector<std::string> results;

    for (int k=0; k < hyps.size(); k++){
        std::string result = token_map[hyps[k][context_size]];

        for (int i=context_size+1; i < hyps[k].size(); i++){
            std::string token = token_map[hyps[k][i]];

            // TODO: recognising '_' is not working
            if (token.at(0) == '_')
                result += " " + token;
            else
                result += token;
        }
        results.push_back(result);
    }
    return results;
}


void print_hyps(std::vector<std::vector<int32_t>> hyps, int context_size = 2){
    std::cout << "Hyps:" << std::endl;
    for (int i=context_size; i<hyps[0].size(); i++)
        std::cout << hyps[0][i] << "-";
    std::cout << "|" << std::endl;
}
