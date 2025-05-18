import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-page',
  imports: [],
  templateUrl: './page.component.html',
  styleUrl: './page.component.css',
})
export class PageComponent {
  constructor(private router: Router) {}

  Function() {
    this.router.navigateByUrl('/main/welcome');
  }
  imageSrc = '/artificial2.png';

  onMouseEnter(): void {
    this.imageSrc = '/artificial.png';
  }

  onMouseLeave(): void {
    this.imageSrc = '/artificial2.png';
  }
}
