(function ($) {
  "use strict";

  function loadComponent(selector, url) {
    return new Promise((resolve, reject) => {
      $(selector).load(url, function (_response, status, xhr) {
        if (status === "error") {
          reject(new Error(`Failed to load ${url}: ${xhr.status}`));
          return;
        }

        resolve();
      });
    });
  }

  $(async function () {
    try {
      await Promise.all([
        loadComponent("#sidebar", "/shared/admin/components/sidebar-component.html"),
        loadComponent("#header", "/shared/admin/components/header-component.html"),
        loadComponent("#footer", "/shared/admin/components/footer-component.html")
      ]);

      await $.getScript("/vendor/admin/app.min.js");
      await $.getScript("/vendor/admin/sidebarmenu.js");
    } catch (error) {
      console.error("Failed to initialise the admin layout:", error);
    }
  });
})(jQuery);
