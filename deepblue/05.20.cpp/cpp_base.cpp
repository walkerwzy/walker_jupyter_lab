#include <iostream>
using namespace std;
#include <list>

int main(){
    list<int> a(5);
    int b[6] = {10, 20, 30, 10, 10, 10};
    a.assign(b, b+5);  // 10, 20, 30, 10, 10
    
    cout << a.size() << endl;  // 5
    a.reverse();  // 10 10 30 20 10
    for (list<int>::iterator i = a.begin(); i!=a.end(); i++){
        cout << *i << " ";  
    }
    cout << "\n";
    a.sort(); // 10 10 10 20 30 
    for (list<int>::iterator i = a.begin(); i!=a.end(); i++){
        cout << *i << " ";
    }
    cout << "\n";    
    a.unique(); // 10 20 30
    for (list<int>::iterator i = a.begin(); i!=a.end(); i++){
        cout << *i << " ";
    }
    cout << "\n";      
}