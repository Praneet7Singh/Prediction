import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class DataService {
  constructor(private http: HttpClient) {}
  Hrt_Pred(data: any) {
    return this.http.post('http://127.0.0.1:8000/cardio_predict', data);
  }
  Dbt_Pred(data: any) {
    return this.http.post('http://127.0.0.1:8000/diabetes_predict', data);
  }
  Alz_Pred(data: any) {
    return this.http.post('http://127.0.0.1:8000/alzheimer_predict', data);
  }
  Sym_Pred(data: any) {
    return this.http.post('http://127.0.0.1:8000/common_predict', data);
  }
  Hrt_Explain(data: any) {
    return this.http.post('http://127.0.0.1:8000/cardio_explain', data);
  }
  Dbt_Explain(data: any) {
    return this.http.post('http://127.0.0.1:8000/diabetes_explain', data);
  }
  Alz_Explain(data: any) {
    return this.http.post('http://127.0.0.1:8000/alzheimer_explain', data);
  }
  Sym_Explain(data: any) {
    return this.http.post('http://127.0.0.1:8000/common_explain', data);
  }
}
