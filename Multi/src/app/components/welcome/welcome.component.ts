import { Component } from '@angular/core';

@Component({
  selector: 'app-welcome',
  imports: [],
  templateUrl: './welcome.component.html',
  styleUrl: './welcome.component.css',
})
export class WelcomeComponent {
  imageSrc = '/artificial2.png';

  onMouseEnter(): void {
    this.imageSrc = '/artificial.png';
  }

  onMouseLeave(): void {
    this.imageSrc = '/artificial2.png';
  }
}
